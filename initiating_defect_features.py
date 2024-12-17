# %%
'''Setup'''
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import ast
from joblib import Parallel, delayed
import seaborn as sns
# import scipy
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import itertools

# %%
'''Functions'''
def process_mask(mask):
    # Find connected components
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


def extract_largest_object(img,points_list):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.array(points_list, dtype=np.int32)],
        255
    )
    img_object, largest_object_mask = process_mask(mask)
    return largest_object_mask

def find_sharpness(numpy_array):
    #Convert array to dataframe
    y,x= np.nonzero(numpy_array)
    df = pd.DataFrame({'x':x,'y':y})
    x0 = df['x'].sum()/df['x'].count()
    y0 = df['y'].sum()/df['y'].count()
    df = df[(df['x']!=0)&(df['y']!=0)].reset_index(drop=True) # to avoid divide by 0 errors later
    #Calculate polar coordinates
    df['x_rel'] = df['x'] - x0
    df['y_rel'] = df['y'] - y0
    df['angle'] = df.apply(lambda row:math.atan2(row['y_rel'],row['x_rel']),axis=1)
    df['distance'] = df.apply(lambda row:math.sqrt(row['y_rel']**2 + row['x_rel']**2),axis=1)
    global_max = df['distance'].max()
    #Find max for each bin
    num_bins = 180
    bin_edges = np.linspace(-math.pi/2, math.pi/2, num_bins + 1)
    bins = pd.IntervalIndex.from_breaks(bin_edges,name='Angle_bin')
    df.index = pd.cut(df['angle'],bins)
    max_df = df.groupby(level=0,observed=False)['distance'].max()
    max_diff = []
    for i in range(0,len(max_df)-1):
        max_diff.append(abs(max_df.iloc[i]-max_df.iloc[i+1])/global_max)
    return max(max_diff) if max_diff else 0

def calculate_aspect_ratio(mask):
    # Find the non-zero mask coordinates
    y_coords, x_coords = np.where(mask > 0)
    
    # Calculate the bounding box dimensions
    height = y_coords.max() - y_coords.min() + 1
    width = x_coords.max() - x_coords.min() + 1
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    return aspect_ratio

def get_perimeter(binary_image):
    # Ensure the image is binary
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If there are contours, return the perimeter of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.arcLength(largest_contour, closed=True)
    
    return 0
    
def calculate_aspect_ratio_rotated(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    (width, height) = rect[1]
    
    # Calculate aspect ratio (ensuring width is always the larger dimension)
    aspect_ratio = max(width, height) / min(width, height)
    
    return aspect_ratio

def multiple_linear_regression(X,Y):
    # Sklearn Linear Regression
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = lr.predict(X_test_scaled)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    return lr, y_pred, mse, r2

def polynomial_regression(X,Y,degree=3):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create polynomial pipeline
    poly_model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.preprocessing.PolynomialFeatures(degree),
        sklearn.linear_model.LinearRegression()
    )

    # Fit model
    poly_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = poly_model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    return poly_model, y_pred, mse, r2

# 3. Ridge Regression
def ridge_regression(X,Y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    # Ridge Regression
    ridge = sklearn.linear_model.Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = ridge.predict(X_test_scaled)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    return ridge, y_pred, mse, r2

# 4. Lasso Regression
def lasso_regression(X,Y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    # Lasso Regression
    lasso = sklearn.linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = lasso.predict(X_test_scaled)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    return lasso, y_pred, mse, r2

def make_feature_df(points_df):
    imgs = []
    energy = []
    stress = []
    laser_power = []
    laser_speed = []
    cycles = []
    sample_id = []
    for row in points_df.iterrows():
        imgs.append(
            extract_largest_object(
                cv2.imread(row[1]['image_path']),
                ast.literal_eval(row[1]['points'])
                )
        )
        laser_power.append(row[1]['scan_power_W'])
        laser_speed.append(row[1]['scan_velocity_mm_s'])
        energy.append(row[1]['energy_density_J_mm3'])
        stress.append(row[1]['test_stress_Mpa'])
        cycles.append(row[1]['cycles'])
        sample_id.append(row[1]['sample_id'])
    portion_of_screen = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / (x.size * x.max()))(x) for x in imgs)
    max_sharpness = Parallel(n_jobs=-1)(delayed(find_sharpness)(x) for x in imgs)
    aspect_ratio = Parallel(n_jobs=-1)(delayed(calculate_aspect_ratio_rotated)(x) for x in imgs)
    perimeter = Parallel(n_jobs=-1)(delayed(get_perimeter)(x) for x in imgs)
    pixels = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / x.max())(x) for x in imgs)
    pixel_perimeter_ratio = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / (x.max() * get_perimeter(x)))(x) for x in imgs)
    data = {
        "sample_id":sample_id,
        "imgs": imgs,
        "screen_portion": portion_of_screen,
        "max_sharpness": max_sharpness,
        "aspect_ratio": aspect_ratio,
        "perimeter": perimeter,
        "pixels": pixels,
        "pixel_perimeter_ratio": pixel_perimeter_ratio,
        "energy_density":energy,
        "stress_Mpa":stress,
        "laser_power":laser_power,
        "laser_speed":laser_speed,
        "cycles": cycles
    }
    return pd.DataFrame(data)

def plot_feature_df(df):
    def annotate_r2(x, y, **kwargs):
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        model = sklearn.linear_model.LinearRegression()
        model.fit(x, y)
        r_squared = model.score(x, y)
        ax = plt.gca()
        ax.annotate(f"$R^2$ = {r_squared:.2f}", xy=(0.05, 0.9), xycoords=ax.transAxes, fontsize=10)

    # Create pairplot with linear regression
    g = sns.pairplot(df, kind="reg", plot_kws={"line_kws": {"color": "red"}})
    # Add R-squared annotations to each plot
    g.map(annotate_r2)
def regression_on_df(df):
    metric_used_list = []
    regression_type = []
    energy_density_list = []
    laser_power_list = []
    scan_speed_list = []
    aspect_ratio_list = []
    sharpness_list = []
    y_pred_list = []
    mse_list = []
    r2_list = []
    for metric in ['log_stress','stress']:
        for regression in [multiple_linear_regression,polynomial_regression,ridge_regression,lasso_regression]:
            for inputs in list(itertools.product([True, False], repeat=5)):
                if True in inputs: # There needs to be at least 1 predictor
                    regression_type.append(regression.__name__)
                    inputs_columns = []
                    if inputs[0]:
                        energy_density_list.append(True)
                        inputs_columns.append('energy_density')
                    else:
                        energy_density_list.append(False)
                    if inputs[1]:
                        laser_power_list.append(True)
                        inputs_columns.append('laser_power')
                    else:
                        laser_power_list.append(False)
                    if inputs[2]:
                        scan_speed_list.append(True)
                        inputs_columns.append('laser_speed')
                    else:
                        scan_speed_list.append(False)
                    if inputs[3]:
                        aspect_ratio_list.append(True)
                        inputs_columns.append('aspect_ratio')
                    else:
                        aspect_ratio_list.append(False)
                    if inputs[4]:
                        sharpness_list.append(True)
                        inputs_columns.append('max_sharpness')
                    else:
                        sharpness_list.append(False)
                    if metric =='log_stress':
                        Y = df['cycles'].apply(math.log)
                        metric_used_list.append('log(stress)')
                    elif metric =='stress':
                        Y = df['cycles'].apply(math.log)*df['stress_Mpa']
                        metric_used_list.append('stress')
                    X = df[inputs_columns]
                    model, y_pred, mse, r2 = regression(X,Y)
                    mse_list.append(mse)
                    r2_list.append(r2)

    regression_dict = {
        "metric_used":metric_used_list,
        "regression_type":regression_type,
        "energy_density":energy_density_list,
        "laser_power":laser_power_list,
        "scan_speed":scan_speed_list,
        "aspect_ratio":aspect_ratio_list,
        "sharpness":sharpness_list,
        "mse":mse_list,
        "r2":r2_list
    }
    df_results = pd.DataFrame(regression_dict)
    return df_results
# %%
if __name__=="__main__":
    print(__name__+" script running")
    # %%
    combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
    points_df = combined_df[~combined_df['points'].isna()]
    print(points_df['image_class'].value_counts())
    print(points_df['sample_id'].value_counts().head(10))
    df = make_feature_df(points_df)
    columns = ["screen_portion","max_sharpness","aspect_ratio","perimeter","pixels"]
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    results_df = regression_on_df(df)
    print(results_df.loc[results_df["r2"].idxmax()])
    print(results_df.loc[results_df[(results_df['aspect_ratio']==False) &(results_df['sharpness']==False)]["r2"].idxmax()])    
    plot = plot_feature_df(df[columns])
else:
    print(__name__+" functions loaded")