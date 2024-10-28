import argparse
parser = argparse.ArgumentParser(description = 'Parameters')
parser.add_argument("epochs",type=int)
parser.add_argument("accumulation_steps",type=int)
parser.add_argument("encoder_pairs",type=int)
parser.add_argument("initial_features",type=int)
parser.add_argument("batch_size",type=int)
parser.add_argument("input_channels",type=int)
parser.add_argument("output_channels",type=int)
parser.add_argument("learning_rate",type=float)
parser.add_argument("imgs_per_transform",type=int)
parser.add_argument("path",type=str)
parser.add_argument('--local-rank', type=int, default=0, help="Local rank of the process for distributed training")
row_structure = '|{:^25}|{:^40}|'
epochs_print = '|{:^5}|{:^5}|{:^10}|{:^10}|{:^10}|'
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
import numpy
import matplotlib.pyplot as plt
import random
import torchvision.transforms.v2 as v2

start_time = time.perf_counter()
torch.manual_seed(9192024)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
# os.environ['find_unused_parameters'] = "True"
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from unet import Unet
from semi_supervised_loss import semi_supervised_loss
from discriminator_loss import discriminator_loss
from multiclass_dataset import Multiclass_dataset
from train_GAN import train_GAN
from FCN import FCDiscriminator
from models_copy.PyTorch.attention_unet import attention_unet

print(torch.cuda.device_count())

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
df = combined_df[(pd.notna(combined_df['path_stitched']))&(pd.notna(combined_df['path_fatigue']))]
x_sup = []
y_sup = []
for csv in df['path_stitched']:
    temp = pd.read_csv(csv)
    x_sup.append(temp['path'].tolist()[0])
for csv in df['path_fatigue']:
    temp = pd.read_csv(csv)
    y_sup.append(temp['path'].tolist()[0])

x_temp = pd.Series(x_sup,name='input')
y_temp = pd.Series(y_sup,name='output')
temp_df = pd.concat([x_temp,y_temp],axis=1)
temp_df.to_csv(args.path + '/dataset.csv')
TRAIN_SPLIT=0.8
VAL_SPLIT=0.2
split_idx=int(len(x_sup)*TRAIN_SPLIT-1)
print('Split idx: '+str(split_idx))
print('Data size: '+str(len(y_sup)))
x_sup_train = [x_sup[:split_idx]]
y_sup_train = [y_sup[:split_idx]]
x_sup_valid = [x_sup[split_idx:]]
y_sup_valid = [y_sup[split_idx:]]

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
    initalization_transform=Resize,
    getitem_transform=blur_affine_trans,
    imgs_per_transform=args.imgs_per_transform
)
valid_ds = Multiclass_dataset(
    x_sup=x_sup_valid,
    y = y_sup_valid,
    initalization_transform=Resize
)

rank = args.local_rank
world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(rank)
setup(rank,world_size)
COMPLETE=False
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate * BATCH_SIZE * world_size
segmentor = attention_unet(input_channels=1,
    output_channels=1,
    initial_features=64
).to(rank)
# Setting the dataset
segmentor = torch.nn.parallel.DistributedDataParallel(segmentor)
train_samp = DistributedSampler(train_ds,rank=rank,shuffle=True)
valid_samp = DistributedSampler(valid_ds,rank=rank,shuffle=False)
while not COMPLETE:
    try:
        train_dl = torch.utils.data.DataLoader(train_ds,sampler=train_samp,batch_size=BATCH_SIZE)
        valid_dl = torch.utils.data.DataLoader(valid_ds,sampler=valid_samp,batch_size=BATCH_SIZE)
        print('dataset finished loading')

        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(lr=LEARNING_RATE,params=segmentor.parameters())
        train_loss = []
        valid_loss = []
        epoch_times = []
        loading_time = time.perf_counter()
        print('Loading time: '+str(loading_time-start_time))
        print(epochs_print.format('Epoch','Rank','Time','Train Loss','Valid Loss',flush=True))
        for i in range(args.epochs):
            segmentor.train()
            temp = []
            only_one=True
            for x,y in train_dl:
                x, y = x.to(rank), y.to(rank)
                if only_one and i==0 and args.local_rank ==0:
                    expected_input = (torch.select(x,0,0).to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    cv2.imwrite(args.path+'/ex_train_x.png',expected_input)
                    expected_output = (torch.select(y,0,0).to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    cv2.imwrite(args.path+'/ex_train_y.png',expected_output)
                    del expected_input
                    del expected_output
                    only_one=False
                dist.barrier() #Make sure all of them are ready
                segmentor.zero_grad()
                s = segmentor(x)
                loss_point = loss(s,y)
                loss_point.backward()
                temp.append(loss_point.item())
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(train_GAN.metrics_list(temp))
            segmentor.eval()
            temp=[]
            only_one=True
            for x,y in valid_dl:
                x, y = x.to(rank), y.to(rank)
                if only_one and i==0 and args.local_rank ==0:
                    expected_input = (torch.select(x,0,0).to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    cv2.imwrite(args.path+'/ex_valid_x.png',expected_input)
                    expected_output = (torch.select(y,0,0).to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    cv2.imwrite(args.path+'/ex_valid_y.png',expected_output)
                    del expected_input
                    del expected_output
                    only_one=False
                s = segmentor(x)
                loss_point=loss(s,y)
                temp.append(loss_point.item())
            valid_loss.append(train_GAN.metrics_list(temp))
            epoch_time= time.perf_counter()
            if i==0:
                epoch_times.append(epoch_time - loading_time)
            else:
                epoch_times.append(epoch_time- sum(epoch_times) - loading_time)
            print(epochs_print.format(str(i),str(rank),str(epoch_times[i])[:10],str(train_loss[i][0])[:10],str(valid_loss[i][0])[:10]),flush=True)
            if args.local_rank==0:
                print('max: '+str(max(epoch_times)))
                print('average: '+str(sum(epoch_times)/len(epoch_times)))
                train_GAN.save_loss_plot(
                    loss_metrics = [
                        train_loss,
                        valid_loss
                    ],
                    legend_titles = [
                        'BCE training loss',
                        'BCE validation loss'
                    ],
                    order = [[0,1]],
                    save_path_fig = args.path+'/loss_figure.png',
                    save_path_csv = args.path+'/loss_figure.csv',
                    subplot_titles=['Loss Figure']
                )
                torch.save(segmentor.state_dict(),args.path+'/model_weights.pt')
        if args.local_rank==0:
            df = combined_df[combined_df['Sample#'].str.contains('CMU9') & -combined_df['path_stitched'].isna()]

            x_unsup=[]
            only_once=True
            for csv in df['path_stitched']:
                if(only_once):
                    temp = pd.read_csv(csv)
                    x_unsup.append(temp['path'].tolist()[0])
                    only_once = False
            x_unsup = [x_unsup]
            ds = Multiclass_dataset(x_unsup=x_unsup,initalization_transform=v2.Resize([512,512],antialias=True))
            only_once = True
            segmentor.eval()
            for x in ds:
                if only_once:
                    x = x.to(rank)
                    x = torch.unsqueeze(x,0)
                    output=segmentor(x)
                    x = torch.squeeze(x,0)
                    output = torch.squeeze(output,0)
                    output=(output.to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    expected_input = (x.to('cpu').permute(1,2,0).detach().numpy()*255).astype(numpy.uint8)
                    expected_input - cv2.cvtColor(expected_input,cv2.COLOR_RGB2BGR)
                    only_once=False
                    cv2.imwrite(args.path+'/ex_out.png',output,)
                    cv2.imwrite(args.path+'/ex_in.png',expected_input)
            COMPLETE=True
            print(row_structure.format('BATCH_SIZE',str(BATCH_SIZE)[-40:]),flush=True)
            print(row_structure.format('LEARNING_RATE',str(LEARNING_RATE)[-40:]),flush=True)
            print(row_structure.format('GPU VRAM size',str(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)[-40:]),flush=True)
    except torch.cuda.OutOfMemoryError as e:
        BATCH_SIZE -=1
        LEARNING_RATE = args.learning_rate * BATCH_SIZE * world_size
        del x
        del y
        torch.cuda.empty_cache()
        print(f"Error occured\n{e}\nOOM error occured. Retrying with a smaller batch size of {BATCH_SIZE}")
        if BATCH_SIZE==0:
            print(f"GPU does not ahve enough memory to support this training scheme.")
            raise torch.cuda.OutOfMemoryError
dist.barrier() #Make sure all of them are ready
dist.destroy_process_group()
# %% Save Script
# Get the path of the current script
script_path = os.path.abspath(__file__)

# Open the script itself and read its contents
with open(script_path, 'r') as script_file:
    script_content = script_file.read()

# Define the path where you want to save the log (e.g., folder 'logs')
log_file_path = args.path + '/script_log.txt'

# Write the content of the script into the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(script_content)

print(f"Script content has been logged to {log_file_path}")
print(1+'1')# Throwing error ends the script