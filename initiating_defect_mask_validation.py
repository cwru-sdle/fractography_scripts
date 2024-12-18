# %%
'''Set Up'''
import math
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import initiating_defect_features
# sys.path.append("pykan/")
from segment_anything import sam_model_registry, SamPredictor
# from pykan.kan import *

# def kan_regression(X,Y)
#     model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
#     dataset = {
#         "train_input":X,
#         "train_label":Y
#     }
#     model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
#     model.


# dataset = {
#     "train_input" = df[['max_sharpness','aspect_ratio','pixel_perimeter_ratio','energy_density']],
#     "train_label" = df.apply(lambda x: x['cycles'].apply(math.log)*x['stress_Mpa'],axis=1)
# }

# output =dataframe_to_tensor(
#     pd.DataFrame(
#         df.apply(lambda x: x['cycles'].apply(math.log)*x['stress_Mpa'],axis=1)))
# inpu = dataframe_to_tensor(
#     df[['max_sharpness','aspect_ratio','pixel_perimeter_ratio','energy_density']]
# )
# dataset = {
#     "train_input":output,
#     "train_label":inpu
# }
# model = KAN(width=[dataset["train_input"].shape[-1],5,dataset['train_label'].shape[-1]], grid=5, k=3, seed=0)
# model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)

def dataframe_to_tensor(df):
    """
    Convert a DataFrame of numerical data into a batched tensor.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing numerical values.
    
    Returns:
        torch.Tensor: Tensor of shape (df.shape[0], df.shape[1]) containing the data from the DataFrame.
    """
    # Ensure that the DataFrame contains only numerical values
    if not df.select_dtypes(include=['number']).shape[1] == df.shape[1]:
        raise ValueError("DataFrame must contain only numerical data.")

    # Convert the DataFrame to a tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)

    return tensor

def find_centroid(numpy_array):
    y,x= np.nonzero(numpy_array)
    df = pd.DataFrame({'x':x,'y':y})
    x0 = df['x'].sum()/df['x'].count()
    y0 = df['y'].sum()/df['y'].count()
    return [x0,y0]

def format_to_SAM(img):
    if(type(img)!=np.ndarray):
        raise TypeError(f"Input to form_to_SAM function must be numpy array, was{type(img)}")
    img = cv2.resize(img,(1024,1024), interpolation = cv2.INTER_AREA)
    if img.ndim == 2:  # Grayscale image
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = np.array(img,dtype=np.uint8)
    return img

def process_row(row):
    try:
        img = format_to_SAM(row['imgs'])
        SAM.set_image(img)
        output = SAM.predict(
            point_coords = np.expand_dims(row['xy'],axis=0),
            point_labels=np.array([1])
        )
        return np.array(output[0]*255,dtype=np.uint8)
    except Exception as e:
        print(row.index)
        print(e)
        return np.nan
def process_mask(mask):
    # Find connected components
    try:
        # print(type(mask))
        # mask = np.array(mask, dtype=np.uint8)
        # print(mask.sum())
        # print(mask.shape)
        mask = np.transpose(mask, (1, 2, 0))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        num_labels, labels = cv2.connectedComponents(mask)
        
        # Count components (excluding background)
        num_objects = num_labels - 1
        
        # Find largest object
        largest_object_mask = np.zeros_like(mask)
        if num_objects > 0:
            # Get unique labels, excluding background (0)
            label_counts = [np.sum(labels == i) for i in range(1, num_labels)]
            largest_object_label = np.argmax(label_counts) + 1
            largest_object_mask = (labels == largest_object_label).astype(np.uint8) * 255
        
        return largest_object_mask
    except Exception as e:
        print(e)
        return np.NAN
def invert_mask(mask):
    """
    Invert a binary mask (0 and 255) by swapping background and foreground.
    Parameters:
        mask (np.ndarray): 2D numpy array representing a binary mask, 
                        expected to have values either 0 or 255.

    Returns:
        np.ndarray: Inverted mask.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")

    # Ensure mask is uint8
    mask = mask.astype(np.uint8, copy=False)

    # Invert the mask
    inverted = 255 - mask
    return inverted
# %%
if __name__=="__main__":
    print(__name__+" script being executed")
    # %%
    '''Load df and SAM'''
    combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
    points_df = combined_df[~combined_df['points'].isna()]
    df = initiating_defect_features.make_feature_df(points_df)
    columns = ["screen_portion","max_sharpness","aspect_ratio","perimeter","pixels"]
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/sam"
    # # sam_checkpoint = path +"/sam_vit_h_4b8939.pth"
    # url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    # sam_checkpoint = urllib.request.urlretrieve(url)

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint="sam_vit_h_4b8939.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only move `sam` to GPU if an NVIDIA GPU is available
    if torch.cuda.is_available():
        sam.to(device=device)
        print("Model moved to GPU.")
    else:
        print("No NVIDIA GPU detected. Using CPU.")
    SAM = SamPredictor(sam)

    np.random.seed(3)
    
    df['xy'] = df['imgs'].apply(find_centroid).apply(np.array)
    SAM_outputs = []
    cross_entropy = []
    best_rows = []


    # %%
    '''Find Best Rows'''
    df['SAM_raw_output'] =df.apply(process_row,axis=1)
    df['SAM_processed_output'] = df['SAM_raw_output'].apply(process_mask).apply(invert_mask)
    df["cross_entropy"] = df.apply(
        lambda x: torch.nn.functional.binary_cross_entropy(
            torch.Tensor(cv2.resize(x['imgs'],(1024,1024))/255),
            torch.Tensor(x['SAM_processed_output']/255)
        )
    ,axis=1).apply(lambda x: x.detach().item())

    for group_string, group in df.groupby(by="sample_id"):
        # print(group_string+" running")
        # group['SAM_output'] =group.apply(process_row,axis=1).apply(process_mask)
        # group["cross_entropy"] = group.apply(
        #     lambda x: torch.nn.functional.binary_cross_entropy(
        #         torch.Tensor(cv2.resize(x['imgs'],(1024,1024))/255),
        #         torch.Tensor(x['SAM_output']/255)
        #     )
        # ,axis=1)
        best_rows.append(
            group.loc[group['cross_entropy'].idxmin()]
        )
        # print(group_string+" ran successfully")
    # best_rows_df = pd.DataFrame(best_rows)
    # print(best_rows_df)
    # best_rows_df.to_csv("best_rows.csv")
    # df['SAM_output']=df.apply(process_row,axis=1).apply(initiating_defect_features.process_mask)
    # df['cross_entropy']=df.apply(lambda x: torch.nn.functional.binary_cross_entropy(x['imgs'],x['SAM_output']),axis=1)
    #  %%
    '''Plotting'''
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, tight_layout=True)
    # ax[0].imshow(cv2.resize(df['imgs'].iloc[0],(1024,1024)))
    # ax[1].imshow(df['SAM_processed_output'].iloc[0])
    # plt.savefig("example_SAM correlation.png")
    # print(df['cross_entropy'].iloc[0])
    # print(f"Input shape: {df['imgs'].iloc[0].shape}")
    # print(f"Output shape: {df['SAM_processed_output'].iloc[0].shape}")
    
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # ax.hist(df['cross_entropy'],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    # # Not as effective in LoF region
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # ax.scatter(df['energy_density'],df['cross_entropy'])

    # # Can't be too big
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # ax.scatter(df['screen_portion'],df['cross_entropy'])

    # # Roughly normally distributed vs stress
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # scatter = ax.scatter(
    #     x=df['stress_Mpa'],
    #     y=df['cross_entropy'],
    #     c=df['energy_density'],
    #     cmap='inferno',  # or another colormap you prefer
    #     vmin=df['energy_density'].min(),
    #     vmax=df['energy_density'].max()/2
    # )

    # # If you want a colorbar:
    # fig.colorbar(scatter, ax=ax, label='energy_density')
# %%
else:
    print(__name__+" functions loaded")