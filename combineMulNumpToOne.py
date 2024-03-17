import os
import numpy as np
import random

# Directory containing the numpy files
directory = "C:\\Users\\mehul\\Downloads\\NumpyDataset\\hin_audios"

def pad_array(array, threshold):
    if len(array) < threshold:
        pad_width = (0, threshold - len(array))
        padded_array = np.pad(array, pad_width, mode='constant')
        return padded_array
    else:
        return array

threshold = 529345

def combine_numpy_files(directory, threshold):
    files = os.listdir(directory)
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sorting based on num
    combined_array = np.empty((0, threshold))
    ct = 0
    for file in files:
        if file.endswith('.npy'):
            data = np.load(os.path.join(directory, file))
            if combined_array.size == 0:
                data = pad_array(data, threshold)
                data = data.reshape(1, -1)  # Reshape data to match the shape of combined_array
                combined_array = np.concatenate((combined_array, data), axis=0)
                print(combined_array.shape)
            else:
                print(ct)
                if(ct>4000):
                    break
                ct+=1
                data = pad_array(data, threshold)
                data = data.reshape(1, -1)  # Reshape data to match the shape of combined_array
                # print(data.shape)
                combined_array = np.concatenate((combined_array, data), axis=0)
    np.save('combined_array.npy', combined_array)



# # Combine numpy files
combine_numpy_files(directory, threshold)

# Load the combined array
combined_array = np.load('combined_array.npy')
print("Shape of combined array:", combined_array.shape)


# # Randomly select 10 files for comparison
# for _ in range(10):
#     random_file = random.choice(os.listdir(directory))
#     if random_file.endswith('.npy'):
#         random_data = np.load(os.path.join(directory, random_file))
#         random_index = random_file.split('_')[-1].split('.')[0]
#         if np.array_equal(random_data, combined_array[random_index]):
#             print(f"Random file {random_file} matches combined array at index {random_index}.")
#         else:
#             print(f"Random file {random_file} does not match combined array at index {random_index}.")

