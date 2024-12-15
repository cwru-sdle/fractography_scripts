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
import scipy.stats 

combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
points_df = combined_df[~combined_df['points'].isna()]
points_df['image_class'].value_counts()
points_df['sample_id'].value_counts().head(10)

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
    #Calculate polar coordinates
    df['x_rel'] = df['x'] - x0
    df['y_rel'] = df['y'] - y0
    df['angle'] = df.apply(lambda row:math.atan(row['y_rel']/row['x_rel']),axis=1)
    df['distance'] = df.apply(lambda row:math.sqrt(row['y_rel']**2 + row['x_rel']**2),axis=1)
    global_max = df['distance'].max()
    #Find max for each bin
    num_bins = 180
    bin_edges = np.linspace(-math.pi/2, math.pi/2, num_bins + 1)
    bins = pd.IntervalIndex.from_breaks(bin_edges,name='Angle_bin')
    df.index = pd.cut(df['angle'],bins)
    max_df = df.groupby(level=0,observed=False)['distance'].max()
    max_diff = []
    for i in range(0,len(max_df)-2):
        max_diff.append(abs(max_df.iloc[i]-max_df.iloc[i+1])/global_max)

    return max(max_diff)
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

# %%
if __name=="main":
# %%
    '''Loading Images and Polygons'''
    imgs = []
    energy = []
    stress = []
    print(points_df.columns)
    for row in points_df.iterrows():
        imgs.append(
            extract_largest_object(
                cv2.imread(row[1]['image_path']),
                ast.literal_eval(row[1]['points'])
                )
        )
        energy.append(row[1]['energy_density_J_mm3'])
        stress.append(row[1]['test_stress_Mpa'])
    # %%
    portion_of_screen = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / (x.size * x.max()))(x) for x in imgs)
    max_sharpness = Parallel(n_jobs=-1)(delayed(find_sharpness)(x) for x in imgs)
    aspect_ratio = Parallel(n_jobs=-1)(delayed(calculate_aspect_ratio_rotated)(x) for x in imgs)
    perimeter = Parallel(n_jobs=-1)(delayed(get_perimeter)(x) for x in imgs)
    pixels = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / x.max())(x) for x in imgs)
    pixel_perimeter_ratio = Parallel(n_jobs=-1)(delayed(lambda x: x.sum() / (x.max() * get_perimeter(x)))(x) for x in imgs)

    data = {
        "screen_portion": portion_of_screen,
        "max_sharpness": max_sharpness,
        "aspect_ratio": aspect_ratio,
        "perimeter": perimeter,
        "pixels": pixels,
        "pixel_perimeter_ratio": pixel_perimeter_ratio,
        "energy_density":energy,
        "stress Mpa":stress
    }
    df = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).dropna()

    def annotate_r2(x, y, **kwargs):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        r_squared = r_value ** 2
        ax = plt.gca()
        ax.annotate(f"$R^2$ = {r_squared:.2f}", xy=(0.05, 0.9), xycoords=ax.transAxes, fontsize=10)

    # Create pairplot with linear regression
    g = sns.pairplot(df, kind="reg", plot_kws={"line_kws": {"color": "red"}})

    # Add R-squared annotations to each plot
    g.map(annotate_r2)
    plt.show()
# %%
else:
    print(__name__+" function loaded")
