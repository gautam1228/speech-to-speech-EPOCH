import numpy as np
import os

def load_numpy_file(filename):
    return np.load(filename)

def check_match(combined_array, loader_array, random_no):
    if loader_array[random_no] is None:
        print(f"Loader array with random number {random_no} is empty.")
        return False
    return np.array_equal(combined_array[random_no], loader_array[random_no])

def main():
    directory = "/path/to/your/directory"
    combined_array = np.load(os.path.join(directory, "combined_array.npy"))
    
    # Initialize an empty array to store loaded arrays
    loader_array = [None] * 10

    for _ in range(10):
        random_no = np.random.randint(0, 100)  # Generate a random number
        filename = os.path.join(directory, f"eng_audio{random_no}.npy")
        if os.path.exists(filename):
            loader_array[random_no] = load_numpy_file(filename)
            print(f"Loaded eng_audio{random_no}.npy")

    for random_no in range(10):
        if check_match(combined_array, loader_array, random_no):
            print(f"Arrays for random number {random_no} match.")
        else:
            print(f"Arrays for random number {random_no} do not match.")

if __name__ == "__main__":
    main()
