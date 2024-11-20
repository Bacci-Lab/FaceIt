import matplotlib.pyplot as plt
import numpy as np
data = np.load(r'C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs\FaceIt\faceit.npz')

motion_energy = data['motion_energy'][:6000]
time = np.linspace(0, len(motion_energy)/30, len(motion_energy))
plt.figure(figsize=(13, 4), facecolor='none')
plt.plot(time, motion_energy, color='darkslateblue')

# Remove frame by hiding top and right spines
ax = plt.gca()
ax.set_facecolor('none')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Remove x and y ticks
ax.set_xticks([])
ax.set_yticks([])

# Add scale bar at the bottom
scalebar_length = 25  # Length of the scale bar in x-axis units
scalebar_y = min(motion_energy) - 1.5  # Adjust this for positioning the scale bar
plt.hlines(y=scalebar_y, xmin=0, xmax=scalebar_length, colors='gray', linewidth=4)  # Add scale bar line
plt.text(scalebar_length / 2, scalebar_y - 6, '25 s', ha='center', fontsize=13)  # Label the scale bar with "25 s"

# Adjust layout
plt.tight_layout()

# Save the figure as an SVG file
plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static/face_motion_plot.svg', format='svg', transparent=True)


# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# frame = [100, 800, 802, 1280]
# # Define the directory containing the numpy array files
# directory_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs"
#
# # List all files in the directory
# files = sorted(os.listdir(directory_path))
#
# # Check if the specific file index exists in the directory
# file_index = 4674  # Specify the array number you want to open
# if file_index < len(files):
#     # Get the file name for the specific index
#     file_name = files[file_index]
#
#     # Construct the full file path
#     file_path = os.path.join(directory_path, file_name)
#
#     # Load the NumPy array
#     current_array = np.load(file_path)
#     print("shape = ", current_array.shape)
#
#     # Extract the region of interest (ROI)
#     current_ROI = current_array[frame[0]:frame[1], frame[2]:frame[3]]
#     print(f"Loaded array from {file_name}:")
#
#     # Plot the image without axes and save it
#     plt.imshow(current_ROI, cmap='gray')  # Use 'gray' for grayscale images
#     plt.axis('off')  # Remove axes and ticks
#     plt.show()

#     # Save the image to a file
#     plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static/second_frame_motion_energy.svg',
#                 format='svg', transparent=True)
#     # output_path = r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\second_frame_motion_energy.png'  # Change the output file name as needed
#     # plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
#
#
# else:
#     print(f"File index {file_index} is out of range. The directory only contains {len(files)} files.")


from pupil_detection import find_ellipse, Image_binarization
# Assuming `find_ellipse` function is already defined
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Define directory and file details
directory_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs"
files = sorted(os.listdir(directory_path))
file_index = 4674  # Specify the array number you want to open

# Check if the specific file index exists
if file_index < len(files):
    file_name = files[file_index]
    file_path = os.path.join(directory_path, file_name)

    # Load the NumPy array
    binary_image = np.load(file_path)

# Get the dimensions of the image
Image_height, Image_width = binary_image.shape

# Create a blank mask with the same dimensions as the binary image
mask = np.zeros_like(binary_image)

# Define the ellipse parameters
center = (660, 160)  # (x, y)
axes = (110, 90)  # (width/2, height/2)
angle = 0
color = 255  # White color to fill the mask
thickness = -1  # Fill the ellipse

# Draw the ellipse on the mask
cv2.ellipse(mask, center, axes, angle, 0, 360, color, thickness)

# Extract the part of the image inside the ellipse
extracted_region = cv2.bitwise_and(binary_image, binary_image, mask=mask)

# Calculate the bounding box coordinates
x_min = max(center[0] - axes[0], 0)
x_max = min(center[0] + axes[0], Image_width)
y_min = max(center[1] - axes[1], 0)
y_max = min(center[1] + axes[1], Image_height)

# Crop the image to the bounding box
cropped_image = extracted_region[y_min:y_max, x_min:x_max]

# Resize the cropped image if necessary
resized_image = cv2.resize(cropped_image, (axes[0] * 2, axes[1] * 2))

# Replace black pixels (value 0) with white (value 255)
resized_image[resized_image == 0] = 255

# Display the original and cropped region using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Cropped and Resized Ellipse Region with White Background")
plt.imshow(resized_image, cmap='gray')

plt.show()


def Image_binarization(chosen_frame_region):
    # Check if the image is already grayscale
    if len(chosen_frame_region.shape) == 2:  # single channel, already grayscale
        sub_region_2Dgray = chosen_frame_region
    else:
        # Convert to grayscale if it has multiple channels (e.g., RGB)
        sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(sub_region_2Dgray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary_image


# Use the function on your resized image
binary_image = Image_binarization(resized_image)

# Call the `find_ellipse` function
ellipse, mean, width, height, angle = find_ellipse(binary_image)

# Plot the binary image with non-zero pixel coordinates
plt.figure(figsize=(10, 5))

# Plot the original binary image
plt.subplot(1, 3, 1)
plt.title('Original Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.scatter(*np.where(binary_image > 0)[::-1], color='red', s=1)  # Plot non-zero pixels
plt.xlabel('x')
plt.ylabel('y')

# Plot the mean center of the ellipse
plt.subplot(1, 3, 2)
plt.title('Ellipse Center')
plt.imshow(binary_image, cmap='gray')
plt.scatter(mean[0], mean[1], color='blue', label='Center', s=50)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

# Plot the fitted ellipse
plt.subplot(1, 3, 3)
plt.title('Fitted Ellipse')
plt.imshow(binary_image, cmap='gray')
ellipse_image = np.zeros_like(binary_image)
cv2.ellipse(ellipse_image, ellipse, 255, 2)
plt.imshow(ellipse_image, cmap='gray', alpha=0.6)
plt.scatter(mean[0], mean[1], color='blue', s=50)  # Center point
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
