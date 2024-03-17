import os
import numpy as np
import random
from multiprocessing import Pool

# Directory containing the numpy files
directory = "C:\\Users\\mehul\\Downloads\\NumpyDataset\\hin_audios"

# Function to read a numpy file
def load_numpy_file(file_path):
    return np.load(file_path)

# Function to read numpy files and combine them into a 2D array
def combine_numpy_files(directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npy')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sorting based on num

    # Preallocate memory for combined array
    total_rows = sum(np.load(file).shape[0] for file in files)
    combined_array = np.empty((total_rows,))  # Assuming all arrays have the same number of columns

    # Load numpy files in parallel
    with Pool() as pool:
        results = pool.map(load_numpy_file, files)

    # Concatenate arrays
    current_row = 0
    ct = 0
    for result in results:
        print(ct)
        ct+=1
        num_rows = result.shape[0]
        combined_array[current_row : current_row + num_rows] = result
        current_row += num_rows

    # Save the combined array
    np.save('combined_array.npy', combined_array)

# Combine numpy files
combine_numpy_files(directory)

# Load the combined array
combined_array = np.load('combined_array.npy')
print("Shape of combined array:", combined_array.shape)

# Randomly select 10 files for comparison
for _ in range(10):
    random_file = random.choice(os.listdir(directory))
    if random_file.endswith('.npy'):
        random_data = np.load(os.path.join(directory, random_file))
        random_index = random_file.split('_')[-1].split('.')[0]
        if np.array_equal(random_data, combined_array[random_index]):
            print(f"Random file {random_file} matches combined array at index {random_index}.")
        else:
            print(f"Random file {random_file} does not match combined array at index {random_index}.")
