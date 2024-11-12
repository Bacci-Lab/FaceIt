import numpy as np
import os
import numpy as np

# Directory containing your .npy files
array_directory = r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images"

# List to store the loaded NumPy arrays
arrays = []

# Iterate over all .npy files in the directory and load them
for filename in os.listdir(array_directory):
    if filename.endswith('.npy'):  # Check if the file is a .npy file
        array_path = os.path.join(array_directory, filename)
        array = np.load(array_path)  # Load the .npy file as a NumPy array
        arrays.append(array)

# Save all arrays into a single .npz file
np.savez_compressed('image_dataset.npz', *arrays)

print("Dataset saved as image_dataset.npz")
