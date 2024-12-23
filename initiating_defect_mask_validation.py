# %%
'''Set Up'''
import math
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn
import initiating_defect_features
from segment_anything import sam_model_registry, SamPredictor
sys.path.append("/mnt/vstor/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/fractography_scripts/pykan/")
from pykan.kan import *

energy_density_label = "Energy Density [J/mm^3]"
FILTER = 0.001

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
def process_mask(mask, kernel_size=3):
    """
    Processes a mask by:
    1. Transposing dimensions (if needed)
    2. Converting to grayscale
    3. Applying a morphological closing to "heal" disconnected parts of the object
    4. Using connected components to find the largest object.

    Parameters:
        mask (np.ndarray): The input mask (assumed shape (C, H, W)).
        kernel_size (int): Size of the kernel used for morphological closing.

    Returns:
        np.ndarray: A binary mask of the largest connected object.
    """
    try:
        # Ensure mask is in the proper shape: (H, W, C)
        mask = np.transpose(mask, (1, 2, 0))
        
        # Convert to a single-channel grayscale image
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Apply morphological closing to "heal" contours
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        healed_mask = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(healed_mask)

        # Count components (excluding background)
        num_objects = num_labels - 1
        
        # Find largest object
        largest_object_mask = np.zeros_like(healed_mask, dtype=np.uint8)
        if num_objects > 0:
            # Get counts for each label (excluding background)
            label_counts = [np.sum(labels == i) for i in range(1, num_labels)]
            largest_object_label = np.argmax(label_counts) + 1
            largest_object_mask = (labels == largest_object_label).astype(np.uint8) * 255
        
        return largest_object_mask
    except Exception as e:
        print(e)
        return np.NaN
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
def plot_energy_density_residuals(data_df, model_df, idx_list, residual_type='MSE'):
    """
    Plot energy density residuals for a list of models and indices.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing the input data.
        model_df (pd.DataFrame): DataFrame containing the models and metadata.
        idx_list (list of tuples): List of (name, index) tuples for the models to plot.
        residual_type (str): Type of residual to compute ('MSE' or 'MAE').
    """

    def generate_random_colors(n):
        """Generate a list of `n` random colors."""
        return [(random.random(), random.random(), random.random()) for _ in range(n)]

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(10, 6))

    for name, idx in idx_list:
        # Extract the selected model and features
        row = model_df.select_dtypes(include=['bool']).iloc[idx]
        row = row[row]
        X = data_df[row.index]

        # Compute the target variable and residuals
        if model_df['metric_used'].iloc[idx] == 'log(stress)':
            Y = data_df['cycles'].apply(math.log) * data_df['stress_Mpa'].apply(math.log)
        elif model_df['metric_used'].iloc[idx] == 'stress':
            Y = data_df['cycles'].apply(math.log) * data_df['stress_Mpa']

        y_pred = model_df.iloc[idx]['model'].predict(X)
        residual = Y - y_pred if residual_type == 'MSE' else abs(Y - y_pred)

        # Scatter plot
        unique_classes = data_df['image_class'].unique()
        color_map = {cls: i for i, cls in enumerate(unique_classes)}  # Map classes to integers
        class_colors = data_df['image_class'].map(color_map)
        ax.scatter(
            x=data_df['energy_density'],
            y=residual,
            c=class_colors,
            cmap=matplotlib.colors.ListedColormap(
                generate_random_colors(len(data_df['image_class'].value_counts().index))
            ),
            alpha=0.6,
            label=f"{name} ({residual_type})"
        )

        # Polynomial regression for residuals
        poly_model, y_pred, mse, r2 = initiating_defect_features.polynomial_regression(
            np.array(data_df['energy_density']).reshape(-1, 1),
            np.array(residual),
            degree=2
        )
        y_pred = poly_model.predict(np.array(data_df['energy_density']).reshape(-1, 1))

        # Access the PolynomialFeatures step
        linear_model = poly_model.named_steps['linearregression']
        poly_features = poly_model.named_steps['polynomialfeatures']
        coefficients = linear_model.coef_
        intercept = linear_model.intercept_
        feature_names = poly_features.get_feature_names_out(['x'])

        # Display the equation
        equation = f"{intercept:.0f}"
        for coef, name in zip(coefficients, feature_names):
            equation += f" + ({coef:.0f})*{name}"

        # Plot the polynomial fit
        sorted_indices = np.argsort(data_df['energy_density'])
        sorted_energy_density = data_df['energy_density'].iloc[sorted_indices]
        sorted_y_pred = y_pred[sorted_indices]
        ax.plot(
            sorted_energy_density,
            sorted_y_pred,
            label=f"{name} Fit: {equation}",
            linewidth=2
        )

    # Finalize the plot
    ax.set_xlabel("Energy Density")
    ax.set_ylabel("Residuals" if residual_type == 'MSE' else "Absolute Residuals")
    ax.legend(title="Models", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def histogram_residuals_by_image_class(data_df, model_df, idx):
    """
    Plot a histogram of residuals grouped by image class on a single graph.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing the input data.
        model_df (pd.DataFrame): DataFrame containing the models and metadata.
        idx (int): Index of the model to use from model_df.
    """
    def generate_random_colors(n):
        """Generate a list of `n` random colors."""
        return [(random.random(), random.random(), random.random()) for _ in range(n)]

    # Extract the selected model and features
    row = model_df.select_dtypes(include=['bool']).iloc[idx]
    row = row[row]
    X = data_df[row.index]

    # Compute the residuals
    if model_df['metric_used'].iloc[idx] == 'log(stress)':
        Y = data_df['cycles'].apply(math.log) * data_df['stress_Mpa'].apply(math.log)
    elif model_df['metric_used'].iloc[idx] == 'stress':
        Y = data_df['cycles'].apply(math.log) * data_df['stress_Mpa']

    y_pred = model_df.iloc[idx]['model'].predict(X)
    residual = Y - y_pred

    # Create the histogram
    unique_classes = data_df['image_class'].unique()
    colors = generate_random_colors(len(unique_classes))

    plt.figure(figsize=(10, 6))
    for img_class, color in zip(unique_classes, colors):
        class_residuals = residual[data_df['image_class'] == img_class]
        plt.hist(
            class_residuals,
            bins=30,
            alpha=0.5,
            label=img_class,
            color=color,
            edgecolor='black'
        )

    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals by Image Class")
    plt.legend(title="Image Class")
    plt.tight_layout()
    plt.show()
# %%
if __name__=="__main__":
    print(__name__+" script being executed")
    # %%
    '''Load df and SAM'''
    combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
    points_df = combined_df[~combined_df['points'].isna()]
    df = initiating_defect_features.make_feature_df(points_df)
    # df = df[df['image_class']=='initiation']
    # columns = ["screen_portion","max_sharpness","aspect_ratio","perimeter","pixels"]
    # df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    # df = df.dropna()

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
    df["inverse_cross_entropy"] = df.apply(
        lambda x: torch.nn.functional.binary_cross_entropy(
            torch.Tensor(cv2.resize(x['imgs'],(1024,1024))/255),
            torch.Tensor(invert_mask(x['SAM_processed_output'])/255)
        )
    ,axis=1).apply(lambda x: x.detach().item())
    df['SAM_processed_output'] = df.apply(lambda x: invert_mask(x['SAM_processed_output']) if x['cross_entropy']>x["inverse_cross_entropy"] else x['SAM_processed_output'],axis=1)
    df['minimum_entropy'] = df.apply(lambda x:min(x['inverse_cross_entropy'],x['cross_entropy']),axis=1)
    #  %%
    '''Show Good Example'''
    good_ex_fig, good_ex_ax = plt.subplots(1, 2, tight_layout=True)
    min_idx = df['cross_entropy'].idxmin()
    good_ex_ax[0].imshow(cv2.resize(df['imgs'].iloc[min_idx],(1024,1024)))
    good_ex_ax[1].imshow(df['SAM_processed_output'].iloc[min_idx])
    good_ex_fig.show()
    print(df['cross_entropy'].iloc[min_idx])
    print(f"Input shape: {df['imgs'].iloc[min_idx].shape}")
    print(f"Output shape: {df['SAM_processed_output'].iloc[min_idx].shape}")
    # %%
    '''Examples best to worst'''
    descending_df = df.sort_values(by='cross_entropy')
    idx=0
    # %%
    idx+=1
    descending_ex_fig, descending_ex_ax = plt.subplots(1, 2, tight_layout=True)
    descending_ex_ax[0].imshow(cv2.resize(descending_df['imgs'].iloc[idx],(1024,1024)))
    descending_ex_ax[1].imshow(descending_df['SAM_processed_output'].iloc[idx])
    print(descending_df['cross_entropy'].iloc[idx])
    print(f"Input shape: {descending_df['imgs'].iloc[idx].shape}")
    print(f"Output shape: {descending_df['SAM_processed_output'].iloc[idx].shape}")
    descending_ex_fig.show()
    # %%
    '''Examples worst to best'''
    ascending_df = df.sort_values(by='minimum_entropy',ascending=False)
    idx=0
    # %%
    idx+=1
    ascending_ex_fig, ascending_ex_ax = plt.subplots(1, 2, tight_layout=True)
    ascending_ex_ax[0].imshow(cv2.resize(ascending_df['imgs'].iloc[idx],(1024,1024)))
    ascending_ex_ax[1].imshow(ascending_df['SAM_processed_output'].iloc[idx])
    print(ascending_df['minimum_entropy'].iloc[idx])
    print(idx)
    ascending_ex_fig.show()
    #  %%
    '''Show Bad Example'''
    bad_ex_fig, bad_ex_ax = plt.subplots(1, 2, tight_layout=True)
    max_idx = df['minimum_entropy'].idxmax()
    bad_ex_ax[0].imshow(cv2.resize(df['imgs'].iloc[max_idx],(1024,1024)))
    bad_ex_ax[1].imshow(df['SAM_processed_output'].iloc[max_idx])
    bad_ex_fig.show()
    print(df['cross_entropy'].iloc[max_idx])
    print(f"Input shape: {df['imgs'].iloc[max_idx].shape}")
    print(f"Output shape: {df['SAM_processed_output'].iloc[max_idx].shape}")

    # %%
    '''Hist Entropy'''    
    hist_fig, hist_ax = plt.subplots(1, 1, tight_layout=True)
    hist_ax.hist(df['minimum_entropy'],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    hist_ax.set_xlabel("Binary Cross Entropy")
    hist_ax.set_ylabel("Mask Count")
    hist_fig.show()
    # %%
    '''screen portion vs cross entropy'''
    # Can't be too big
    pve_fig, (pve_ax,good_pve_ax,bad_pve_ax) = plt.subplots(3, 1, tight_layout=True,figsize=(16,20))
    pve_scatter = pve_ax.scatter(df['screen_portion'],df['minimum_entropy'],
        c=df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    pve_ax.set_xlabel("Portion of Screen")
    pve_ax.set_ylabel("Binary Cross Entropy")
    print(df['minimum_entropy'].apply(lambda x:x<2).value_counts().rename("Enrtopy<2"))
    good_df = df[df['minimum_entropy'].apply(lambda x:x<2)].copy()
    good_pve_scatter = good_pve_ax.scatter(good_df['screen_portion'],good_df['minimum_entropy'],
        c=good_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    good_pve_ax.set_xlabel("Portion of Screen")
    good_pve_ax.set_ylabel("Binary Cross Entropy")
    # Compute a linear fit using numpy's polyfit
    good_x = good_df['screen_portion'].values
    good_y = good_df['minimum_entropy'].values
    good_m, good_b = np.polyfit(good_x, good_y, 1)
    good_y_pred = good_m * good_x + good_b
    good_y_mean = np.mean(good_y)
    good_ss_tot = np.sum((good_y - good_y_mean)**2)
    good_ss_res = np.sum((good_y - good_y_pred)**2)
    good_r_squared = 1 - (good_ss_res / good_ss_tot)
    # Plot the linear regression line
    good_pve_ax.plot(good_x, good_y_pred, color='red', label=f'{round(good_m,1)}*x+{round(good_b,1)}, R^2={round(good_r_squared,3)}')
    bad_df = df[df['minimum_entropy'].apply(lambda x:x>2)].copy()
    bad_pve_scatter = bad_pve_ax.scatter(bad_df['screen_portion'],bad_df['minimum_entropy'],
        c=bad_df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2)
    bad_pve_ax.set_xlabel("Portion of Screen")
    bad_pve_ax.set_ylabel("Binary Cross Entropy")
    # Compute a linear fit using numpy's polyfit
    bad_x = bad_df['screen_portion'].values
    bad_y = bad_df['minimum_entropy'].values
    bad_m, bad_b = np.polyfit(bad_x, bad_y, 1)
    bad_y_pred = bad_m * bad_x + bad_b
    bad_y_mean = np.mean(bad_y)
    bad_ss_tot = np.sum((bad_y - bad_y_mean)**2)
    bad_ss_res = np.sum((bad_y - bad_y_pred)**2)
    bad_r_squared = 1 - (bad_ss_res / bad_ss_tot)
    bad_m, bad_b = np.polyfit(bad_df['screen_portion'], bad_df['minimum_entropy'], 1)
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
    entropy_vs_energy_scatter = entropy_vs_energy_ax.scatter(df['energy_density'],df['cross_entropy'],
        c=df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max())
    entropy_vs_energy_ax.set_xlabel("Energy Density [J/mm^3]")
    entropy_vs_energy_ax.set_ylabel("Binary Cross Entropy")
    entropy_vs_energy_fig.colorbar(entropy_vs_energy_scatter, ax=entropy_vs_energy_ax, label=energy_density_label)
    entropy_vs_energy_fig.show()
    # %%
    '''Stress vs entropy'''
    # Roughly normally distributed vs stress
    stress_vs_entropy_fig, stress_vs_entropy_ax = plt.subplots(1, 1, tight_layout=True)
    stress_vs_entropy_scatter = stress_vs_entropy_ax.scatter(
        x=df['stress_Mpa'],
        y=df['cross_entropy'],
        c=df['energy_density'],
        cmap='inferno',  # or another colormap you prefer
        vmin=df['energy_density'].min(),
        vmax=df['energy_density'].max()/2
    )
    stress_vs_entropy_ax.set_xlabel("Stress[MPa]")
    stress_vs_entropy_ax.set_ylabel("Binary Cross Entropy")
    stress_vs_entropy_fig.colorbar(stress_vs_entropy_scatter, ax=stress_vs_entropy_ax, label=energy_density_label)
    stress_vs_entropy_fig.show()
    # %%
    '''Screen Portion Filter'''
    # Can still be pretty small
    passes_column = df['screen_portion'].apply(lambda x:x>.001)
    print("Passing Filter: "+str(passes_column.sum()))
    passes_df = df[passes_column].reset_index(drop=True)
    plt.imshow(passes_df.iloc[passes_df['screen_portion'].idxmin()]['SAM_processed_output'])
    print("Scrren Portion minimum: "+str(df['screen_portion'].min()))
    # %%
    '''no na regression'''
    process_features = ['energy_density', 'laser_power', 'laser_speed']
    image_features = ['aspect_ratio', 'max_sharpness','pixel_perimeter_ratio']
    features = process_features + image_features
    regressions = [
        initiating_defect_features.multiple_linear_regression,
        initiating_defect_features.polynomial_regression,
        initiating_defect_features.ridge_regression,
        initiating_defect_features.lasso_regression,
        # KAN_regression
        ]
    no_na_df = df.dropna()
    no_na_results_df = initiating_defect_features.regression_on_df(no_na_df,
        regression_function_list=regressions,
        features=features)
    print(no_na_results_df.loc[no_na_results_df["r2"].idxmax()])
    print(no_na_results_df.loc[no_na_results_df[no_na_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])    

    # %%
    '''Regression Good Dataframe'''
    good_df = df[df.apply(lambda x: (x['minimum_entropy']<2) & (x['screen_portion']>FILTER),axis=1)].dropna()
    good_results_df = initiating_defect_features.regression_on_df(good_df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(good_df)
    print(good_results_df.loc[good_results_df["r2"].idxmax()])
    print(good_results_df.loc[good_results_df[good_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])    
    # %%
    '''Regression Bad Dataframe'''
    bad_df = df[df.apply(lambda x: x['cross_entropy']>2,axis=1)].dropna()
    bad_results_df = initiating_defect_features.regression_on_df(bad_df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(good_df)
    print(bad_results_df.loc[bad_results_df["r2"].idxmax()])
    print(bad_results_df.loc[bad_results_df[bad_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])    

    # %%
    '''Middling Dataframe'''
    bad_df = df[df.apply(lambda x:((x['cross_entropy']<5) & (x['cross_entropy']>0.2)),axis=1)].dropna()
    bad_results_df = initiating_defect_features.regression_on_df(bad_df,
        regression_function_list=regressions,
        features=features)
    # initiating_defect_features.plot_feature_df(good_df)
    print(bad_results_df.loc[bad_results_df["r2"].idxmax()])
    print(bad_results_df.loc[bad_results_df[bad_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])    
    # %%
    '''Filter Dataframe'''
    filter_df = df[df['screen_portion']>FILTER].dropna().reset_index(drop=True)
    filter_results_df = initiating_defect_features.regression_on_df(filter_df,
        regression_function_list=regressions,
        features=features)
    print(filter_results_df.loc[filter_results_df["r2"].idxmax()])
    print(filter_results_df.loc[filter_results_df[filter_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])    
    # %%
    '''Images Features Improve Results'''
    filter_df = df[(df['screen_portion']>FILTER)&(df['minimum_entropy']<2)].dropna().reset_index(drop=True)
    filter_results_df = initiating_defect_features.regression_on_df(filter_df,
        regression_function_list=regressions,
        features=features)
    print("Remaing Image: "+str(filter_df.shape[0]))
    print(filter_results_df.loc[filter_results_df["r2"].idxmax()])
    print(filter_results_df.loc[filter_results_df[filter_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()])
    print(filter_results_df.loc[filter_results_df[filter_results_df[process_features].eq(False).all(axis=1)]["r2"].idxmax()])
    print(filter_df['image_class'].value_counts())
    histogram_residuals_by_image_class(filter_df, filter_results_df, idx=filter_results_df["r2"].idxmax())
    MSE_energy_density(filter_df,filter_results_df,idx=filter_results_df["r2"].idxmax())
    # %%
    '''Might Have to Do with Image Size'''
    filter_df = df[(df['screen_portion']>FILTER)&(df['minimum_entropy']<2)].dropna().reset_index(drop=True)
    best_rows = []
    for group_string, group in filter_df.groupby(by="sample_id"):
        best_rows.append(
            group.loc[group['screen_portion'].idxmin()]
        )
    filter_df = pd.DataFrame(best_rows).dropna().reset_index(drop=True)
    filter_results_df = initiating_defect_features.regression_on_df(filter_df,
        regression_function_list=regressions,
        features=features)
    print("Remaining Images: "+str(filter_df.shape[0]))
    all_features_best_idx = filter_results_df["r2"].idxmax()
    image_features_best_idx = filter_results_df[filter_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()
    process_features_best_idx = filter_results_df[filter_results_df[process_features].eq(False).all(axis=1)]["r2"].idxmax()
    idxs = [("All Features",all_features_best_idx),("Image Features",image_features_best_idx),("Process Features",process_features_best_idx)]
    print(filter_results_df.iloc[all_features_best_idx])
    print(filter_results_df.iloc[image_features_best_idx])
    print(filter_results_df.iloc[process_features_best_idx])
    for idx in idxs:
        histogram_residuals_by_image_class(filter_df, filter_results_df, idx=idx[1])
    plot_energy_density_residuals(filter_df,filter_results_df,idx_list=idxs,residual_type='MSE')
    plot_energy_density_residuals(filter_df,filter_results_df,idx_list=idxs,residual_type='MAE')
    # %%
    '''Exploration'''
    filter_df = df[(df['screen_portion']>FILTER)&(df['minimum_entropy']<2)].dropna().reset_index(drop=True)
    best_rows = []
    for group_string, group in filter_df.groupby(by="sample_id"):
        best_rows.append(
            group.loc[group['screen_portion'].idxmax()]
        )
    filter_df = pd.DataFrame(best_rows).dropna().reset_index(drop=True)
    filter_results_df = initiating_defect_features.regression_on_df(filter_df,
        regression_function_list=regressions,
        features=features)
    print("Remaining Images: "+str(filter_df.shape[0]))
    all_features_best_idx = filter_results_df["r2"].idxmax()
    image_features_best_idx = filter_results_df[filter_results_df[image_features].eq(False).all(axis=1)]["r2"].idxmax()
    process_features_best_idx = filter_results_df[filter_results_df[process_features].eq(False).all(axis=1)]["r2"].idxmax()
    print(filter_results_df.iloc[all_features_best_idx])
    print(filter_results_df.iloc[image_features_best_idx])
    print(filter_results_df.iloc[process_features_best_idx])
    histogram_residuals_by_image_class(filter_df, filter_results_df, idx=filter_results_df["r2"].idxmax())
    MSE_energy_density(filter_df,filter_results_df,idx=filter_results_df["r2"].idxmax())
# %%
else:
    print(__name__+" functions loaded")