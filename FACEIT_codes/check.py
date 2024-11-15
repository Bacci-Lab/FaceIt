import numpy as np

# Path to the .npz file
file_path = r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images/faceit.npz'

# Load the .npz file
data = np.load(file_path)

# List all the keys (array names) in the file
print("Available data keys:", data.files)

# Access specific data arrays
pupil_center = data['pupil_center']
pupil_area = data['pupil_dilation']
motion_energy = data['motion_energy']

# Print or use the data as needed
print("Pupil Center:", pupil_center)
print("Pupil Dilation:", pupil_area)
print("Motion Energy:", motion_energy)

# Close the .npz file (optional but good practice)
data.close()
