import pandas as pd
import numpy as np

# Change paths when needed
scene1_csv = './result_#.csv'
scene2_csv = './test_result.csv'

# Read the csv files
data1 = pd.read_csv(scene1_csv)
data2 = pd.read_csv(scene2_csv)

# Merge the datasets 
merged_data = pd.merge(data1, data2, on=['object_name', 'vertex_index'], suffixes=('_scene1', '_scene2'))

# Compute the squared differences for x, y, z coordinates
merged_data['squared_error'] = (
    (merged_data['x_scene1'] - merged_data['x_scene2'])**2 +
    (merged_data['y_scene1'] - merged_data['y_scene2'])**2 +
    (merged_data['z_scene1'] - merged_data['z_scene2'])**2
)

# RMSE
rmse = np.sqrt(merged_data['squared_error'].mean())
print(f"RMSE between the scenes: {rmse:.4f}")