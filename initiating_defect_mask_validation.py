# %%
'''Set Up'''
import math
import sys
import numpy as np
import pandas as ps
import cv2
import torch
sys.path.append("/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/fractography_scripts")
import initiating_defect_features
sys.path.append("/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/pykan")
from kan import *
from segment_anything import sam_model_registry, SamPredictor

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
def process_row(row):
    try:
        img = row['imgs']
        img = cv2.resize(img,(1024,1024), interpolation = cv2.INTER_AREA)
        if img.ndim == 2:  # Grayscale image
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        SAM.set_image(img)
        output = SAM.predict(
            point_coords = np.expand_dims(row['xy'],axis=0),
            point_labels=np.array([1])
        )
        return output[0]
    except Exception as e:
        print(row.index)
        print(e)
        return False
def process_mask(mask):
    # Find connected components
    mask = np.array(mask, dtype=np.uint8)
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
    
    return num_objects, largest_object_mask

# %%
if __name__=="__main__":
    print(__name__+" script being executed")
    # %%
    combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
    points_df = combined_df[~combined_df['points'].isna()]
    df = initiating_defect_features.make_feature_df(points_df)
    columns = ["screen_portion","max_sharpness","aspect_ratio","perimeter","pixels"]
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/sam"
    sam_checkpoint = path +"/sam_vit_h_4b8939.pth"
    # url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    # sam_checkpoint = urllib.request.urlretrieve(url)

    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

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
    for group_string, group in df.groupby(by="sample_id"):
        try:
            print(group_string+" running")
            group['temp'] =df.apply(process_row,axis=1)     
            print("row processed")       
            group['SAM_output'] =df.apply(process_mask)
            print("SAM output found")
            group["cross_entropy"] = df.apply(lambda x: torch.nn.functional.binary_cross_entropy(x['imgs'],x['SAM_output']),axis=1)
            print("cross entropy found")
            best_rows.append(
                group.loc[group['cross_entropy'].idxmin()]
            )
            print(group_string+" ran successfully")
        except Exception as e:
            print(f"Error in group {group_string}: {e}")
    best_rows_df = pd.DataFrame(best_rows)
    print(best_rows_df)
    best_rows.to_csv("best_rows.csv")
    # df['SAM_output']=df.apply(process_row,axis=1).apply(initiating_defect_features.process_mask)
    # df['cross_entropy']=df.apply(lambda x: torch.nn.functional.binary_cross_entropy(x['imgs'],x['SAM_output']),axis=1)
    #  %%
    
# %%
else:
    print(__name__+" functions loaded")