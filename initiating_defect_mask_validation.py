# %%
'''Set Up'''
import math
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import sklearn
import initiating_defect_features
from segment_anything import sam_model_registry, SamPredictor
sys.path.append("/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/fractography_scripts/pykan/")
from pykan.kan import *

energy_density_label = "Energy Density [J/mm^3]"

def KAN_regression(X,Y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    model = KAN(width=[X_test.shape[1],X_test.shape[1],1],k=3,device=device)
    X_train = torch.Tensor(X_train.values).to(device)
    X_test = torch.Tensor(X_test.values).to(device)
    y_train = torch.Tensor(y_train.values).to(device)
    y_test = torch.Tensor(y_test.values).to(device)
    data = {
        'train_input':X_train,
        'test_input':X_test,
        'train_label':y_train,
        'test_label':y_test
    }
    model.fit(dataset=data, opt="LBFGS", steps=100, lamb=0.01, reg_metric='edge_forward_spline_n')
    def r2_score(y_true, y_pred):
        # Ensure that tensors are 1-dimensional if necessary
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        # Compute the residual sum of squares
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        # Compute the total sum of squares (proportional to variance of the data)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        mse = ss_res / y_true.size(0)

        # Compute R²
        # Add a small epsilon in the denominator if there's a chance of division by zero.
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
        
        return r2, mse
    y_pred = model(data['test_input'])
    r2, mse = r2_score(data['test_label'],y_pred)
    return model, y_pred.to("cpu").detach().numpy(), r2.item(), mse.item()

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

def process_row(img, xy, model):
    try:
        img = format_to_SAM(img)
        model.set_image(img)
        output = model.predict(
            point_coords = np.expand_dims(xy,axis=0),
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
    best_rows = []


    # %%
    '''Find Cross Entropy'''
    df['SAM_raw_output'] =df.apply(lambda x: process_row(x['imgs'],x['xy'],SAM),axis=1)
    df['SAM_processed_output'] = df['SAM_raw_output'].apply(process_mask).apply(invert_mask)
    df["cross_entropy"] = df.apply(
        lambda x: torch.nn.functional.binary_cross_entropy(
            torch.Tensor(cv2.resize(x['imgs'],(1024,1024))/255),
            torch.Tensor(x['SAM_processed_output']/255)
        )
    ,axis=1).apply(lambda x: x.detach().item())
    # %%
    '''Find Best Rows'''
    for group_string, group in df.groupby(by="sample_id"):
        best_rows.append(
            group.loc[group['cross_entropy'].idxmin()]
        )
    best_rows_df = pd.DataFrame(best_rows)
    # best_rows_df.to_csv("best_rows.csv")
    # #  %%
    # '''Show Good Example'''
    # good_ex_fig, good_ex_ax = plt.subplots(1, 2, tight_layout=True)
    # min_idx = df['cross_entropy'].idxmin()
    # good_ex_ax[0].imshow(cv2.resize(df['imgs'].iloc[min_idx],(1024,1024)))
    # good_ex_ax[1].imshow(df['SAM_processed_output'].iloc[min_idx])
    # good_ex_fig.show()
    # print(df['cross_entropy'].iloc[min_idx])
    # print(f"Input shape: {df['imgs'].iloc[min_idx].shape}")
    # print(f"Output shape: {df['SAM_processed_output'].iloc[min_idx].shape}")
    # #  %%
    # '''Show Bad Example'''
    # bad_ex_fig, bad_ex_ax = plt.subplots(1, 2, tight_layout=True)
    # max_idx = df['cross_entropy'].idxmax()
    # bad_ex_ax[0].imshow(cv2.resize(df['imgs'].iloc[max_idx],(1024,1024)))
    # bad_ex_ax[1].imshow(df['SAM_processed_output'].iloc[max_idx])
    # bad_ex_fig.show()
    # print(df['cross_entropy'].iloc[max_idx])
    # print(f"Input shape: {df['imgs'].iloc[max_idx].shape}")
    # print(f"Output shape: {df['SAM_processed_output'].iloc[max_idx].shape}")

    # %%
    '''Hist Entropy'''    
    hist_fig, hist_ax = plt.subplots(1, 1, tight_layout=True)
    hist_ax.hist(df['cross_entropy'],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    hist_fig.show()
    # %%
    '''screen portion vs cross entropy'''
    # Can't be too big
    pve_fig, (pve_ax,good_pve_ax,bad_pve_ax) = plt.subplots(3, 1, tight_layout=True,figsize=(16,20))
    pve_scatter = pve_ax.scatter(df['screen_portion'],df['cross_entropy'],
        c=df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    pve_ax.set_xlabel("Portion of Screen")
    pve_ax.set_ylabel("Binary Cross Entropy")
    print(df['cross_entropy'].apply(lambda x:x<2).value_counts().rename("Enrtopy<2"))
    good_df = df[df['cross_entropy'].apply(lambda x:x<2)].copy()
    good_pve_scatter = good_pve_ax.scatter(good_df['screen_portion'],good_df['cross_entropy'],
        c=good_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    good_pve_ax.set_xlabel("Portion of Screen")
    good_pve_ax.set_ylabel("Binary Cross Entropy")
    # Compute a linear fit using numpy's polyfit
    good_x = good_df['screen_portion'].values
    good_y = good_df['cross_entropy'].values
    good_m, good_b = np.polyfit(good_x, good_y, 1)
    good_y_pred = good_m * good_x + good_b
    good_y_mean = np.mean(good_y)
    good_ss_tot = np.sum((good_y - good_y_mean)**2)
    good_ss_res = np.sum((good_y - good_y_pred)**2)
    good_r_squared = 1 - (good_ss_res / good_ss_tot)
    # Plot the linear regression line
    good_pve_ax.plot(good_x, good_y_pred, color='red', label=f'{round(good_m,1)}*x+{round(good_b,1)}, R^2={round(good_r_squared,3)}')
    bad_df = df[df['cross_entropy'].apply(lambda x:x>2)].copy()
    bad_pve_scatter = bad_pve_ax.scatter(bad_df['screen_portion'],bad_df['cross_entropy'],
        c=bad_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    bad_pve_ax.set_xlabel("Portion of Screen")
    bad_pve_ax.set_ylabel("Binary Cross Entropy")
    # Compute a linear fit using numpy's polyfit
    bad_x = bad_df['screen_portion'].values
    bad_y = bad_df['cross_entropy'].values
    bad_m, bad_b = np.polyfit(bad_x, bad_y, 1)
    bad_y_pred = bad_m * bad_x + bad_b
    bad_y_mean = np.mean(bad_y)
    bad_ss_tot = np.sum((bad_y - bad_y_mean)**2)
    bad_ss_res = np.sum((bad_y - bad_y_pred)**2)
    bad_r_squared = 1 - (bad_ss_res / bad_ss_tot)
    bad_m, bad_b = np.polyfit(bad_df['screen_portion'], bad_df['cross_entropy'], 1)
    bad_pve_ax.plot(bad_x, bad_y_pred, color='red', label=f'{round(bad_m,1)}*x+{round(bad_b,1)},R^2={round(bad_r_squared,3)}')
    bad_pve_ax.legend()
    pve_fig.colorbar(bad_pve_scatter, ax=bad_pve_ax, label=energy_density_label)
    good_pve_ax.legend()
    pve_fig.colorbar(good_pve_scatter, ax=good_pve_ax, label=energy_density_label)
    good_pve_ax.legend()
    pve_ax.plot(good_x, good_y_pred, color='red', label=f'{round(good_m,1)}*x+{round(good_b,1)},R^2={round(good_r_squared,3)}')
    pve_ax.plot(bad_x, bad_y_pred, color='red', label=f'{round(bad_m,1)}*x+{round(bad_b,1)},R^2={round(bad_r_squared,3)}')
    pve_ax.legend()
    pve_fig.colorbar(pve_scatter, ax=pve_ax, label=energy_density_label)
    pve_fig.show()
    # %%
    '''entropy vs energy density'''
    # Not as effective outside of process region
    entropy_vs_energy_fig, entropy_vs_energy_ax = plt.subplots(1, 1, tight_layout=True)
    entropy_vs_energy_scatter = entropy_vs_energy_ax.scatter(good_df['energy_density'],good_df['cross_entropy'],
        c=good_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=good_df['energy_density'].min(),
        vmax=good_df['energy_density'].max())
    entropy_vs_energy_ax.set_xlabel("Energy Density [J/mm^3]")
    entropy_vs_energy_ax.set_ylabel("Binary Cross Entropy")
    entropy_vs_energy_fig.colorbar(entropy_vs_energy_scatter, ax=entropy_vs_energy_ax, label=energy_density_label)
    entropy_vs_energy_fig.show()
    # %%
    '''Stress vs entropy'''
    # Roughly normally distributed vs stress
    stress_vs_entropy_fig, stress_vs_entropy_ax = plt.subplots(1, 1, tight_layout=True)
    stress_vs_entropy_scatter = stress_vs_entropy_ax.scatter(
        x=good_df['stress_Mpa'],
        y=good_df['cross_entropy'],
        c=good_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=good_df['energy_density'].min(),
        vmax=good_df['energy_density'].max()/2
    )
    stress_vs_entropy_ax.set_xlabel("Stress[MPa]")
    stress_vs_entropy_ax.set_ylabel("Binary Cross Entropy")
    stress_vs_entropy_fig.colorbar(stress_vs_entropy_scatter, ax=stress_vs_entropy_ax, label=energy_density_label)
    stress_vs_entropy_fig.show()
    # %%
    '''Regression All Images'''
    process_features = ['energy_density', 'laser_power', 'laser_speed']
    image_features = ['aspect_ratio', 'max_sharpness','pixel_perimeter_ratio']
    features = process_features + image_features
    columns = ["screen_portion","max_sharpness","aspect_ratio","perimeter","pixels",'cross_entropy']
    regressions = [initiating_defect_features.multiple_linear_regression,
        initiating_defect_features.polynomial_regression,
        initiating_defect_features.ridge_regression,
        initiating_defect_features.lasso_regression,
        KAN_regression
        ]
    
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    all_results_df = initiating_defect_features.regression_on_df(df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(df)
    print(all_results_df.loc[all_results_df["r2"].idxmax()])
    print(all_results_df.loc[all_results_df[(all_results_df['aspect_ratio']==False) &(all_results_df['max_sharpness']==False)]["r2"].idxmax()])    

    # %%
    '''Regression Good Dataframe'''
    good_df[columns] = good_df[columns].replace([np.inf, -np.inf], np.nan)
    good_df = good_df.dropna()
    good_results_df = initiating_defect_features.regression_on_df(good_df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(good_df)
    print(good_results_df.loc[good_results_df["r2"].idxmax()])
    print(good_results_df.loc[good_results_df[(good_results_df['aspect_ratio']==False) &(good_results_df['max_sharpness']==False)]["r2"].idxmax()])    

    # %%
    '''Regression best rows'''
    good_best_df = best_rows_df[best_rows_df['cross_entropy'].apply(lambda x:x<2)]
    good_best_df[features] = good_best_df[features].replace([np.inf, -np.inf], np.nan)
    good_best_df = good_best_df.dropna()
    good_best_df_results = initiating_defect_features.regression_on_df(
        good_best_df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(good_df)
    print(good_best_df_results.loc[good_best_df_results["r2"].idxmax()])
    print(good_best_df_results.loc[good_best_df_results[(good_best_df_results['aspect_ratio']==False) &(good_best_df_results['max_sharpness']==False)&(good_best_df_results['pixel_perimeter_ratio']==False)]["r2"].idxmax()])    

# %%
else:
    print(__name__+" functions loaded")