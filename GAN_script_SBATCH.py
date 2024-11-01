import argparse
parser = argparse.ArgumentParser(description = 'Parameters')
parser.add_argument("epochs",type=int)
parser.add_argument("accumulation_steps",type=int)
parser.add_argument("encoder_pairs",type=int)
parser.add_argument("initial_features",type=int)
parser.add_argument("batch_size",type=int)
parser.add_argument("input_channels",type=int)
parser.add_argument("output_channels",type=int)
parser.add_argument("learning_rate_seg",type=float)
parser.add_argument("learning_rate_disc",type=float)
parser.add_argument("imgs_per_transform",type=int)
parser.add_argument("path",type=str)
parser.add_argument('--local-rank', type=int, default=0, help="Local rank of the process for distributed training")
row_structure = '|{:^25}|{:^40}|'
epochs_print = '|{:^5}|{:^5}|{:^10}|'
args = parser.parse_args()
for arg, value in vars(args).items():
    print(row_structure.format(arg,str(value)[-40:]),flush=True)
print('-'*(25+40+3))
# %%
import pandas as pd
import os
import cv2
import time
import sys
import torch
import torchvision.transforms.v2 as v2

start_time = time.perf_counter()
torch.manual_seed(9192024)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
os.environ['find_unused_parameters'] = "True"
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from unet import Unet
from semi_supervised_loss import semi_supervised_loss
from discriminator_loss import discriminator_loss
from multiclass_dataset import Multiclass_dataset
from train_GAN import train_GAN
from FCN import FCDiscriminator
from models_copy.PyTorch.attention_unet import attention_unet

print(torch.cuda.device_count())


# Setting the dataset
combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
df = combined_df[(pd.notna(combined_df['path_stitched']))&(pd.notna(combined_df['path_fatigue']))]
x_sup = []
y_sup = []
x_unsup = []
for csv in df['path_stitched']:
    temp = pd.read_csv(csv)
    x_sup.append(temp['path'].tolist()[0])
for csv in df['path_fatigue']:
    temp = pd.read_csv(csv)
    y_sup.append(temp['path'].tolist()[0])
df = combined_df[(pd.notna(combined_df['path_stitched']))&(-pd.notna(combined_df['path_fatigue']))]
for csv in df['path_stitched']:
    temp = pd.read_csv(csv)
    x_unsup.append(temp['path'].tolist()[0])
x_temp = pd.Series(x_sup,name='input')
y_temp = pd.Series(y_sup,name='output')
temp_df = pd.concat([x_temp,y_temp],axis=1)
temp_df.to_csv(args.path + '/dataset_supervised.csv')
pd.DataFrame(pd.Series(x_unsup,name='input')).to_csv(args.path + '/dataset_unsupervised.csv')

TRAIN_SPLIT=0.8
VAL_SPLIT=0.2
split_idx=int(len(x_sup)*TRAIN_SPLIT-1)
print('Split idx: '+str(split_idx))
print('Data size: '+str(len(y_sup)))
x_sup_train = [x_sup[:split_idx]]
y_sup_train = [y_sup[:split_idx]]
x_sup_valid = [x_sup[split_idx:]]
y_sup_valid = [y_sup[split_idx:]]
x_unsup = [x_unsup] # x_unsup is not modified
blur_affine_trans = v2.Compose(
    [
        v2.RandomAffine(
            degrees=180,
            scale=[0.5,2],
            shear=[-15,-15,15,15]
        ),
        v2.GaussianBlur(
            [3,3],
        )
    ]
)
unblur_affine_trans = v2.Compose(
    [
        v2.RandomAffine(
            degrees=180,
            scale=[0.5,2],
            shear=[-15,-15,15,15]
        ),
    ]
)
Resize = v2.Resize([512,512],antialias=True)
train_ds = Multiclass_dataset(
    x_sup=x_sup_train,
    y = y_sup_train,
    initalization_transform=[
        Resize
    ],
    getitem_transform=blur_affine_trans,
    imgs_per_transform=args.imgs_per_transform
)
valid_ds = Multiclass_dataset(
    x_sup=x_sup_valid,
    y = y_sup_valid,
    initalization_transform=Resize,
)
print('Unseg Dataset finished loading')
unseg_ds = Multiclass_dataset(
    x_unsup=x_unsup[:len(x_sup_train)*args.imgs_per_transform],
    initalization_transform=[
        Resize
    ],
    getitem_transform=blur_affine_trans,
    imgs_per_transform=1
)
print('Unseg Dataset finished loading')
rank = args.local_rank
world_size = int(os.environ['WORLD_SIZE'])
device = 'cuda:'+str(rank)
print(device)
print('initalizing process group')
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.cuda.empty_cache()

BATCH_SIZE = args.batch_size
LEARNING_RATE_SEG = args.learning_rate_seg * BATCH_SIZE * world_size
LEARNING_RATE_DISC = args.learning_rate_disc * BATCH_SIZE * world_size
segmentor = attention_unet(
    input_channels=1,
    output_channels=1,
    initial_features=64
).to(rank)
segmentor = torch.nn.parallel.DistributedDataParallel(segmentor)
discriminator = FCDiscriminator(args.output_channels).to(rank)
discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)

train_samp = torch.utils.data.distributed.DistributedSampler(train_ds,rank=rank,shuffle=True)
valid_samp = torch.utils.data.distributed.DistributedSampler(valid_ds,rank=rank,shuffle=False)
unseg_samp = torch.utils.data.distributed.DistributedSampler(unseg_ds,rank=rank,shuffle=True)
COMPLETE = False
while not COMPLETE:
    try:
        print('Attempt to load')
        train_dl = torch.utils.data.DataLoader(train_ds,batch_size=args.batch_size,sampler=train_samp)
        valid_dl = torch.utils.data.DataLoader(valid_ds,batch_size=args.batch_size,sampler=valid_samp)
        unseg_dl = torch.utils.data.DataLoader(unseg_ds,batch_size=args.batch_size,sampler=unseg_samp)
        print('dataset finished loading')

        # %%
        # Setting the loss functions
        class Seg_gen_loss():
            def __init__(self,w_adv):
                self.w_adv = w_adv
                self.BCE = torch.nn.BCELoss()
                self.adv_loss = discriminator_loss()
            def forward(self, seg_input, mask,disc_input):
                
                return self.BCE(seg_input,mask) + self.w_adv*self.adv_loss(disc_input,True)
        seg_gen_loss = Seg_gen_loss(0.15)
        class Unseg_gen_loss():
            def __init__(self,w_adv,w_semi):
                self.w_adv = w_adv
                self.w_semi = w_semi
                self.adv_loss = discriminator_loss()
                self.semi_loss = semi_supervised_loss()
            def forward(self,disc_input,seg_input):
                return self.w_adv * self.adv_loss(disc_input,True) + self.w_semi * self.semi_loss(disc_input,seg_input)
        unseg_gen_loss = Unseg_gen_loss(0.3,0.3)
        gen_optimizer = torch.optim.Adam(lr=LEARNING_RATE_SEG,params=segmentor.parameters())
        disc_optimizer = torch.optim.Adam(lr=LEARNING_RATE_DISC,params=discriminator.parameters())
        
        class train_GAN_intermed_save(train_GAN):
            def adversarial_learning(self,save_path=None,save_epochs=1):
                self.raw_disc_loss = []
                self.seg_disc_loss = []
                self.sup_seg_loss = []
                self.unsup_seg_loss = []
                self.val_seg_loss = []
                start_time = time.time()
                print('Time: '+str(time.time()-start_time))
                for epoch in range(self.epochs):
                    #Supervised
                    print(f'start seg training epoch {epoch}: '+str(time.time()-start_time))
                    self.segmented_training_epoch()

                    #Unsupervised
                    print(f'start unseg training epoch {epoch}: '+str(time.time()-start_time))
                    self.unsegmented_training_epoch()
                    
                    #Validation epoch
                    print(f'start validation epoch {epoch}: '+str(time.time()-start_time))
                    self.validation_epoch()
                    if save_path!=None and epoch%save_epochs==0:
                        self.save_loss_plot(
                                    loss_metrics=[
                                        self.raw_disc_loss,
                                        self.seg_disc_loss,
                                        self.sup_seg_loss,
                                        self.unsup_seg_loss,
                                        self.val_seg_loss,
                                    ],
                                    subplot_titles=['Training Loss','Validation Loss'],
                                    legend_titles = [
                                        'raw_disc_loss',
                                        'seg_disc_loss',
                                        'sup_seg_loss',
                                        'unsup_seg_loss',
                                        "val_seg_loss",
                                    ],
                                    order=self.order,
                                    save_path_fig=save_path+'/figure.png',
                                    save_path_csv=save_path+'/figure,csv'
                        )
                        torch.save(obj=self.seg_model.state_dict(),f = save_path+'/model.pt')
        training_scheme = train_GAN_intermed_save(
            disc_loss=discriminator_loss(),
            seg_gen_loss=seg_gen_loss.forward,
            unseg_gen_loss=unseg_gen_loss.forward,
            seg_model=segmentor,
            disc_model=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            seg_train_dl=train_dl,
            seg_val_dl=valid_dl,

            unseg_train_dl=unseg_dl,
            epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            device = device,
            order=[[0,1],[2,3,4]],
        )
        training_scheme.adversarial_learning(save_path=args.path,save_epochs=2)
        COMPLETE=True
    except torch.cuda.OutOfMemoryError as e:
        BATCH_SIZE -=1
        LEARNING_RATE_SEG = args.learning_rate_seg * BATCH_SIZE * world_size
        LEARNING_RATE_DISC = args.learning_rate_disc * BATCH_SIZE * world_size
        del x
        del y
        torch.cuda.empty_cache()
        print(f"Error occured\n{e}\nOOM error occured. Retrying with a smaller batch size of {BATCH_SIZE}")
        if BATCH_SIZE==0:
            print(f"GPU does not ahve enough memory to support this training scheme.")
            raise torch.cuda.OutOfMemoryError
dist.destroy_process_group()

# Get the path of the current script
script_path = os.path.abspath(__file__)

# Open the script itself and read its contents
with open(script_path, 'r') as script_file:
    script_content = script_file.read()

# Define the path where you want to save the log (e.g., folder 'logs')
log_folder = args.path + '/script_log.txt'

# Write the content of the script into the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(script_content)

print(f"Script content has been logged to {log_file_path}")
