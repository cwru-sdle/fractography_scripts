
# %%
'''Points'''
import pandas as pd
import cv2
import ast
import numpy as np
import math
import scipy
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

def calculate_filled_portion(mask):
    """
    Calculate the portion of the image filled by objects in a mask.
    
    Parameters:
    mask (np.ndarray): A 2D or 3D binary mask where objects are represented by non-zero values.
    
    Returns:
    float: The fraction of the image area covered by objects in the mask.
    """
    mask = np.array(mask)
    if not isinstance(mask, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if mask.ndim not in (2, 3):
        raise ValueError("Mask must be a 2D or 3D array.")
    
    # Count non-zero pixels in the mask
    filled_area = np.count_nonzero(mask)
    
    # Calculate the total area of the image
    total_area = mask.size  # Total number of pixels
    
    # Calculate the fraction
    filled_portion = filled_area / total_area
    
    return filled_portion

combined_df = pd.read_csv('/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/combined_df.csv')
points_df = combined_df[~combined_df['points'].isna()]
# print(points_df['image_class'].value_counts())
# print(points_df['sample_id'].value_counts().head(10))
sample_id = []
basename = []
numb_of_objects = []
largest_object_masks = []
for i, (group_string, group) in enumerate(points_df.drop_duplicates(subset='image_basename').groupby("sample_id")):
    basename.extend(group['image_basename'])
    # if i >= STOP:
    #     break
    imgs = group['image_path'].apply(cv2.imread).tolist()
    points_list = group['points'].apply(ast.literal_eval).tolist()

    masks = []
    for j in range(len(imgs)):
        mask = np.zeros(imgs[j].shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [np.array(points_list[j], dtype=np.int32)],
            255
        )
        img_object_count, largest_object_mask = process_mask(mask)

        largest_object_masks.append(largest_object_mask)
        numb_of_objects.append(img_object_count)
        sample_id.append(group_string)
    
series = [
    [sample_id,'sample_id'],
    [basename,'image_basename'],
    [numb_of_objects,'num_of_objects'],
    [largest_object_masks,'largest_object_masks']
]

df = pd.concat(
    list(map(lambda x: pd.Series(x[0],name=x[1]),series)),axis=1)

points_df = pd.merge(
    combined_df[['sample_id','scan_power_W','scan_velocity_mm_s','energy_density_J_mm3','image_basename']].drop_duplicates(subset='image_basename'),
    df,
    on='image_basename')
# points_df['largest_object_masks'] = points_df['largest_object_masks'].astype(np.ndarray)
# I selected for only the largest object
# for group_string, group in points_df.groupby('image_basename'):
#     if len(group['num_of_objects']):
#         print(group['num_of_objects'].mean())

portions = []
group_strings = []
for group_string, group in points_df.groupby('image_basename'):
    # Apply the function and store the result in 'portions'
    portion_values = group['largest_object_masks'].apply(lambda x: calculate_filled_portion(np.array(x)))
    
    # Append the result to the list
    portions.append(portion_values)
    group_strings.append(group['energy_density_J_mm3'])
# %%
x_SAM_process = []
y_SAM_process = []
x_SAM_test = []
y_SAM_test = []
x_not_SAM_process = []
y_not_SAM_process = []
x_not_SAM_test = []
y_not_SAM_test = []
for group_string, group in combined_df.groupby('sample_id'):
    no_points = group[(group['image_class']=='initiation') & (group['points'].isna())]
    points = group[(group['image_class']=='initiation') & (~group['points'].isna())]
    if len(no_points.index)>=1:
        x_not_SAM_process.append(no_points['scan_velocity_mm_s'].iloc[0])
        y_not_SAM_process.append(no_points['scan_power_W'].iloc[0])
        x_not_SAM_test.append(no_points['cycles'].iloc[0])
        y_not_SAM_test.append(no_points['test_stress_Mpa'].iloc[0])
    elif  len(points.index)>=1:
        x_SAM_process.append(points['scan_velocity_mm_s'].iloc[0])
        y_SAM_process.append(points['scan_power_W'].iloc[0])
        x_SAM_test.append(points['cycles'].iloc[0])
        y_SAM_test.append(points['test_stress_Mpa'].iloc[0])
plt.figure(figsize=(10, 5))

# Process Variables Histograms
plt.subplot(1, 2, 1)
plt.hist(x_not_SAM_process, bins='auto', color='red', alpha=0.7, label='No Points')
plt.hist(x_SAM_process, bins='auto', color='blue', alpha=0.7, label='With Points')
plt.xlabel('Scan Velocity (mm/s)')
plt.ylabel('Frequency')
plt.title('Process Variables')
plt.legend()

plt.subplot(2, 2, 1)
plt.hist(y_not_SAM_process, bins='auto', color='red', alpha=0.7, label='No Points')
plt.hist(y_SAM_process, bins='auto', color='blue', alpha=0.7, label='With Points')
plt.xlabel('Scan Velocity (mm/s)')
plt.ylabel('Frequency')
plt.title('Process Variables')
plt.legend()

# Test Variables Histograms
plt.subplot(1, 2, 2)
plt.hist(x_not_SAM_test, bins='auto', color='red', alpha=0.7, label='No Points')
plt.hist(x_SAM_test, bins='auto', color='blue', alpha=0.7, label='With Points')
plt.xlabel('Cycles')
plt.ylabel('Frequency')
plt.title('Test Variables')
plt.legend()
plt.tight_layout()
plt.show()


def jitter(arr, jitter_amount=5):
    return arr + np.random.uniform(-jitter_amount, jitter_amount, len(arr))

plt.figure(figsize=(10, 5))

# Process Variables Plot
plt.subplot(1, 2, 1)
plt.hist(jitter(x_not_SAM_process), jitter(y_not_SAM_process), color='red', label='No Points', alpha=0.7)
plt.hist(jitter(x_SAM_process), jitter(y_SAM_process), color='blue', label='With Points', alpha=0.7)
plt.xlabel('Scan Velocity (mm/s)')
plt.ylabel('Scan Power (W)')
plt.title('Process Variables')
plt.legend()

# Test Variables Plot
plt.subplot(1, 2, 2)
plt.hist(jitter(x_not_SAM_test), jitter(y_not_SAM_test), color='red', label='No Points', alpha=0.7)
plt.hist(jitter(x_SAM_test), jitter(y_SAM_test), color='blue', label='With Points', alpha=0.7)
plt.xlabel('Cycles')
plt.ylabel('Test Stress (MPa)')
plt.title('Test Variables')
plt.legend()

plt.tight_layout()
plt.show()

# %%
SAM_process = []
SAM_test = []
not_SAM_process = []
not_SAM_test = []
for group_string, group in combined_df.groupby('sample_id'):
    no_points = group[(group['image_class']=='initiation') & (group['points'].isna())]
    points = group[(group['image_class']=='initiation') & (~group['points'].isna())]
    if not (len(no_points.index)>=1 and len(points.index)>=1):
        if len(no_points.index)>=1:
            not_SAM_process.append(no_points['energy_density_J_mm3'].iloc[0])
        elif  len(points.index)>=1:
            SAM_process.append(points['energy_density_J_mm3'].iloc[0])

plt.figure(figsize=(10, 5))

# Process Variables Histograms
plt.subplot(1, 2, 1)
plt.hist(x_not_SAM_process, bins='auto', color='red', alpha=0.7, label='No Points')
plt.hist(x_SAM_process, bins='auto', color='blue', alpha=0.7, label='With Points')
plt.xlabel('Energy Desnity [J/mm^3]')
plt.ylabel('Frequency')
plt.title('Process Variables')
plt.legend()

# %%
# for group_string, group in points_df.groupby('image_basename'):
#     # Apply the function and store the result in 'portions'
#     portion_values = group['largest_object_masks'].apply(lambda x: calculate_filled_portion(np.array(x)))    
#     portions.append(portion_values)
#     group_strings.append(group['energy_density_J_mm3'])

# %%
x_SAM_test = []
x_not_SAM_test = []
for group_string, group in combined_df.groupby('sample_id'):
    no_points = group[(group['image_class']=='initiation') & (group['points'].isna())]
    points = group[(group['image_class']=='initiation') & (~group['points'].isna())]
    if not (len(no_points.index)>=1 and len(points.index)>=1):
        if len(no_points.index)>=1:
            cycles =no_points['cycles'].iloc[0]
            stress =no_points['test_stress_Mpa'].iloc[0]
            x_not_SAM_test.append(math.log(cycles)*math.log(stress))
        elif  len(points.index)>=1:
            cycles =points['cycles'].iloc[0]
            stress =points['test_stress_Mpa'].iloc[0]
            x_SAM_test.append(math.log(cycles)*math.log(stress))
plt.figure(figsize=(10, 5))

# Test Variables Histograms
plt.subplot(1, 2, 1)
plt.hist(x_not_SAM_test, bins='auto', color='red', alpha=0.7, label='No Points')
plt.hist(x_SAM_test, bins='auto', color='blue', alpha=0.7, label='With Points')
plt.xlabel('log(cycles)*los(stress [Mpa])')
plt.ylabel('Frequency')
plt.title('Testing Variables')
plt.legend()
plt.show()

x_not_SAM_test = np.array(x_not_SAM_test)
x_SAM_test = np.array(x_SAM_test)
x_not_SAM_test = x_not_SAM_test[~np.isnan(x_not_SAM_test)]
x_SAM_test = x_SAM_test[~np.isnan(x_SAM_test)]

statistic, p_value = scipy.stats.ks_2samp(np.array(x_not_SAM_test),np.array(x_SAM_test))

print("P value: "+str(p_value))
if p_value < 0.05:
    print("Reject the null hypothesis: distributions are different")
else:
    print("Fail to reject the null hypothesis: distributions are the same")

# %%
success = 0
failure = 1
for group_string, group in combined_df.groupby('sample_id'):
    no_points = group[(group['image_class']=='initiation') & (group['points'].isna())]
    points = group[(group['image_class']=='initiation') & (~group['points'].isna())]
    if len(no_points.index)>=1:
        success+=1
    elif len(points.index)>=1:
        failure+=1
print("Success rate of: "+str(success/(success+failure)))
