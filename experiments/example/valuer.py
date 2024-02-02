import os
import pickle

# Read the logfile.txt
with open('logfile_depth.txt', 'r') as file:
    lines = file.readlines()

# Base directory
base_dir = '../../data/KITTI/object/training/values_depth'

# For each line, get the filename and the float value
for line in lines:
    filename, value = line.strip().split()
    value = float(value)

    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Save the float value to the .pkl file in the base directory
    with open(os.path.join(base_dir, f'{filename}.pkl'), 'wb') as file:
        pickle.dump(value, file)