import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pupil_detection import Image_binarization
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
save_directory = r'C:\Users\faezeh.rabbani\Documents\save_paper_figure'
directory_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs"
# directory_path = r"C:\Users\faezeh.rabbani\Desktop\FaceCamera-imgs"
# file_index = 64
file_index = 11475
def load_image(directory_path, file_index):

    files = sorted(os.listdir(directory_path))
    # Check if the specific file index exists
    if file_index < len(files):
        file_name = files[file_index]
        file_path = os.path.join(directory_path, file_name)
        binary_image = np.load(file_path)
    return binary_image


def plot_simplefigure(image, title , save_directory = None, file_name = None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title, fontsize = 18)
    plt.axis("off")
    # if save_directory:
    #     save_svg(save_directory, file_name)
    plt.show()

def find_ellipse(binary_image):
    """
    Fits an ellipse to non-zero pixels in a binary image and visualizes eigenvalues and eigenvectors.

    Parameters:
    binary_image (np.ndarray): A binary image where non-zero pixels are considered for fitting.

    Returns:
    tuple: A tuple containing:
        - ellipse ((tuple), (tuple), float): Center (x, y), axes lengths (width, height), and rotation angle in degrees.
        - mean (tuple): Center of the fitted ellipse (x, y).
        - width (float): Width of the fitted ellipse (major axis length).
        - height (float): Height of the fitted ellipse (minor axis length).
        - angle (float): Rotation angle of the fitted ellipse in radians.
    """
    # Get coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary_image > 0))
    coords = coords[:, [1, 0]]

    # Return default values if no coordinates are found
    if coords.size == 0:
        return ((0, 0), (0, 0), 0), (0.0, 0.0), 0.0, 0.0, 0.0

    # Calculate mean position and center the coordinates
    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean


    # Compute covariance matrix and its eigenvalues and eigenvectors
    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the ellipse properties
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])
    #
    # width = np.sqrt(2) * np.sqrt(eigenvalues[0])
    # height = np.sqrt(2) * np.sqrt(eigenvalues[1])

    # Create the ellipse representation
    ellipse = (int(mean[0]), int(mean[1])), (int(width * 2), int(height * 2)), np.degrees(angle)

    # Visualization
    plt.figure(figsize=(7, 5))
    plt.title("Application of PCA on pupil data", fontsize = 12)
    # Plot data points
    plt.scatter(coords[:, 0], coords[:, 1], color='mediumorchid', s=4, label="pupil pixels")

    # Plot eigenvectors scaled by eigenvalues
    for i in range(2):
        vector = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2  # Scale by eigenvalues
        plt.quiver(mean[0], mean[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1,
                   color=['darkcyan', 'orange'][i], label=f"PC {i+1} ")

    # Add legend and axis labels
    plt.legend(frameon=False)

    # Optionally, remove spines and ticks for a clean look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')  # Ensure equal scaling
    plt.grid(True)
    plt.gca().invert_yaxis()
    # plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\PCA_application.svg',
    #             format='svg', transparent=True)
    plt.close()
    return ellipse, (float(mean[0]), float(mean[1])), width, height, angle
def Image_binarization(chosen_frame_region, threshold = 240):
    # Check if the image is already grayscale
    if len(chosen_frame_region.shape) == 2:  # single channel, already grayscale
        sub_region_2Dgray = chosen_frame_region
    else:
        # Convert to grayscale if it has multiple channels (e.g., RGB)
        sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(sub_region_2Dgray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def make_binary_image(resized_image):
    binary_image = Image_binarization(resized_image)
    #Visualize the original and binary images
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image, cmap='gray')
    plt.axis('off')
    plt.title("Original Image", pad=20)

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.title("Binary Image", pad=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\Original_VS_binary.svg',
                format='svg', transparent=True)
    plt.close()
def plot_non_zero_cordinates(non_zero_coords):
    plt.figure(figsize=(4, 4), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_facecolor('none')  # Set axis background to transparent
    plt.title("Non-zero Pixel Coordinates")
    plt.scatter(non_zero_coords[:, 1], non_zero_coords[:, 0], s=1, color='mediumpurple')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\Non-zero_Pixel.png',
                transparent=True)
    plt.close()
def plot_clusters(non_zero_coords, labels):
    # Visualize the clustered data
    plt.figure(figsize=(4, 4), facecolor='none')
    ax2 = plt.subplot(1, 1, 1)
    ax2.set_facecolor('none')
    plt.title("Image Clustered")
    plt.scatter(non_zero_coords[:, 1], non_zero_coords[:, 0], c=labels, cmap='tab20b', s=1)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
    plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\Image_clustered.png',
                 transparent=True)
    plt.close()
def save_svg(save_directory_path, file_name):
    file_name = file_name + ".svg"
    file_path = os.path.join(save_directory_path, file_name)
    plt.savefig(file_path,
                format='svg', transparent=True)

def find_cluster(binary):
    """
    Detects the largest connected component (assumed pupil) in a binary image
    and returns a convex-hull-filled binary mask of that cluster.

    Parameters:
    binary_image (np.ndarray): A binary image where non-zero pixels represent the object.
    mnd: (unused here, kept for compatibility)

    Returns:
    np.ndarray: A binary mask with the largest detected cluster filled in.
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    # If only background found, return blank image
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    # Ignore label 0 (background), find the largest component
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(sizes)

    # Create mask for the largest component
    pupil_mask = np.zeros_like(binary, dtype=np.uint8)
    pupil_mask[labels == largest_label] = 255

    # Find contours of the largest component
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(binary, dtype=np.uint8)

    # Merge all contour points
    all_points = np.vstack(contours)

    # Compute convex hull
    hull = cv2.convexHull(all_points)

    # Create final mask and draw convex hull
    hull_image = np.zeros_like(binary)
    cv2.drawContours(hull_image, [hull], -1, 255, -1)
    return hull_image


# def find_cluster(resized_image):
    # """
    # Detects all clusters of non-zero pixels in a binary image and returns an image
    # highlighting all detected clusters. Visualizes intermediate steps.
    #
    # Parameters:
    # resized_image (np.ndarray): An input image to be binarized and processed.
    #
    # Returns:
    # np.ndarray: A binary image with all clusters highlighted.
    # """
    # binary_image = Image_binarization(resized_image)
    # non_zero_coords = np.column_stack(np.where(binary_image > 0))
    #
    # if non_zero_coords.shape[0] == 0:
    #     return np.zeros_like(binary_image, dtype=np.uint8)
    #
    # clustering = DBSCAN(eps=6, min_samples=1).fit(non_zero_coords)
    # labels = clustering.labels_
    #
    # # Plotting all clusters with different colors
    # # plt.figure(figsize=(8, 8))
    # # unique_labels = np.unique(labels)
    # # colors = plt.cm.get_cmap('tab20', len(unique_labels))
    # #
    # # for label in unique_labels:
    # #     cluster_coords = non_zero_coords[labels == label]
    # #     plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0],
    # #                 s=5, color=colors(label), label=f'Cluster {label + 1}')
    # #
    # # plt.title("All Detected Clusters")
    # # plt.gca().invert_yaxis()
    # # plt.axis('equal')
    # # # plt.axis('off')
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # # plt.tight_layout()
    # # plt.show()
    #
    # # Create an output image highlighting all clusters
    # # detected_clusters = np.zeros_like(binary_image, dtype=np.uint8)
    # # for point in non_zero_coords:
    # #     cv2.circle(detected_clusters, (point[1], point[0]), 1, (255,), -1)
    #
    # # Find the biggest cluster
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # biggest_cluster_label = unique_labels[np.argmax(counts)]  # Label of the largest cluster
    #
    # # Extract only the largest cluster points
    # biggest_cluster_coords = non_zero_coords[labels == biggest_cluster_label]
    # ###############
    # boundary_thickness = 1
    # # Create an empty image and draw the biggest cluster
    # biggest_cluster_image = np.zeros_like(binary_image, dtype=np.uint8)
    # for point in biggest_cluster_coords:
    #     cv2.circle(biggest_cluster_image, (point[1], point[0]), 1, (255,), -1)
    #
    # # # Find the contours (edges) of the biggest cluster
    # # contours, _ = cv2.findContours(biggest_cluster_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # surroundings_image = np.zeros_like(binary_image, dtype=np.uint8)
    # #
    # # # Draw the contour to extract surroundings
    # # cv2.drawContours(surroundings_image, contours, -1, (255,), thickness=boundary_thickness)
    # #
    # # surroundings_coords = np.column_stack(np.where(surroundings_image > 0))
    # #
    # # # Plot only the biggest cluster
    # # plt.figure(figsize=(8, 8))
    # # plt.scatter(biggest_cluster_coords[:, 1], biggest_cluster_coords[:, 0],
    # #             s=5, color='red', label=f'Biggest Cluster ({biggest_cluster_label})')
    # #
    # # ################
    # # plt.scatter(surroundings_coords[:, 1], surroundings_coords[:, 0],
    # #             s=5, color='blue', label="Surroundings of Biggest Cluster")
    # # ####################
    # #
    # # plt.title("Biggest Detected Cluster")
    # # plt.gca().invert_yaxis()
    # # plt.axis('equal')
    # # plt.axis('off')
    # # plt.legend()
    # # plt.show()
    #
    # # Create an output image highlighting only the biggest cluster
    # biggest_cluster_image = np.zeros_like(binary_image, dtype=np.uint8)
    # for point in biggest_cluster_coords:
    #     cv2.circle(biggest_cluster_image, (point[1], point[0]), 1, (255,), -1)
    # ###################################
    # # surroundings_cluster_image = np.zeros_like(binary_image, dtype=np.uint8)
    # # for point in surroundings_coords:
    # #     cv2.circle(surroundings_cluster_image, (point[1], point[0]), 1, (255,), -1)
    # #
    # # plt.figure(figsize=(8, 8))
    # # plt.title("Compare surroundings_cluster_image")
    # # plt.imshow(surroundings_cluster_image)
    # # plt.figure(figsize=(8, 8))
    # # plt.title("Compare biggest_cluster_image")
    # # plt.imshow(biggest_cluster_image)
    # # plt.show()
    #
    # ######################
    # binary_image = biggest_cluster_image
    #
    # # # Find contours to determine the object size
    # # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # largest_contour = max(contours, key=cv2.contourArea)  # Get the largest detected shape
    # # # Get bounding box dimensions
    # # x, y, w, h = cv2.boundingRect(largest_contour)
    # #
    # # print("x, y, w, h",x, y, w, h)
    # #
    # # # Define an adaptive kernel size based on the object size
    # # kernel_size = max(w, h)
    # # kernel_size = max(5, kernel_size)  # Ensure a minimum size
    # #
    # # # Create an elliptical kernel
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # #
    # # # Apply morphological closing
    # # closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # #
    # # # Display the original and closed images
    # # plt.figure(figsize=(10, 5))
    # #
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(binary_image, cmap='gray')
    # # plt.title("Original Binary Image")
    # # plt.axis("off")
    # #
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(closed_image, cmap='gray')
    # # plt.title(f"After Morphological Closing (Kernel: {kernel_size}x{kernel_size})")
    # # plt.axis("off")
    # #
    # # plt.show()
    #
    # #################################
    #
    # # Find all contours
    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Merge all contour points into a single array
    # all_points = np.vstack(contours)  # Stack all contour points together
    #
    # # Compute the convex hull
    # hull = cv2.convexHull(all_points)
    #
    # # Create a blank image to draw the convex hull
    # hull_image = np.zeros_like(binary_image)
    #
    # # Draw and fill the convex hull
    # cv2.drawContours(hull_image, [hull], -1, 255, -1)  # -1 fills the hull with white
    #
    # # Display results
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(binary_image, cmap='gray')
    # plt.title("Original Binary Image")
    # plt.axis("off")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(hull_image, cmap='gray')
    # plt.title("After Convex Hull (Forced Circular Shape)")
    # plt.axis("off")
    #
    # plt.show()
    #
    # return hull_image



def crop_Image(binary_image, center,angle, axes):
    mask = np.zeros_like(binary_image)
    thickness = -1  # Fill the ellipse
    color = 255  # White color to fill the mask

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

    return cropped_image, resized_image

def define_ellipse_parameters():
    # Define the ellipse parameters
    center = (620, 140)  # (x, y)
    axes = (100, 70)  # (width/2, height/2)
    angle = 0
    return center, axes, angle

def plot_ROI_VS_realImage(binary_image, resized_image):
    # Display the original and cropped region using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(binary_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Cropped and Resized Ellipse Region with White Background")
    plt.imshow(resized_image, cmap='gray')
    plt.close()

def overlap_reflect(reflections, pupil_ellipse, binary_image):
    """
    Modifies the binary image by setting overlapping regions between the pupil ellipse
    and reflection ellipses to white (255).

    Parameters:
    reflections (list or None): A list of tuples defining reflection ellipses.
    pupil_ellipse (tuple): A tuple defining the pupil ellipse (center, axes, angle).
    binary_image (np.ndarray): The input binary image to be modified.

    Returns:
    np.ndarray: The modified binary image with overlapping regions highlighted.
    """
    # Proceed only if reflections are provided
    if reflections is not None:
        # Create black masks for the pupil and reflections
        pupil_mask = np.zeros_like(binary_image, dtype=np.uint8)
        reflection_mask = np.zeros_like(binary_image, dtype=np.uint8)

        # Draw the pupil ellipse on the pupil mask
        cv2.ellipse(pupil_mask, pupil_ellipse, color = 255, thickness = -1)

        # Draw each reflection ellipse on the reflection mask
        for reflection in reflections:
            cv2.ellipse(reflection_mask, reflection, color = 255, thickness = -1)

        # Identify overlapping regions between the pupil and reflection masks
        overlap_mask = cv2.bitwise_and(pupil_mask, reflection_mask)

        # Find the coordinates of the overlapping pixels
        overlap_coords = np.column_stack(np.where(overlap_mask > 0))

        # Update the binary image at the overlapping coordinates
        binary_image[overlap_coords[:, 0], overlap_coords[:, 1]] = 255

    # Return the modified binary image
    return binary_image
#
# ########################################################
reflect1_ellipse = ((93, 63), (30, 27), np.float64(86.59735845915252))
reflect1_ellipse2 = ((87, 84), (15, 15), np.float64(86.59735845915252))
all_reflects = [reflect1_ellipse, reflect1_ellipse2]
binary_image = load_image(directory_path, file_index)
center, axes, angle = define_ellipse_parameters()
Image_height, Image_width = binary_image.shape
cropped_image, resized_image = crop_Image(binary_image, center,angle, axes)
########################
# Convert to float for precise calculations
alpha = 3  # Contrast control (1.0 = original, <1.0 decreases, >1.0 increases)
beta = 0  # Brightness control

# Apply contrast and brightness adjustments
adjusted = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)

# Display the result
cv2.imshow("Original", resized_image)
cv2.imshow("Contrast Adjusted", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
##################################################

detected_cluster = find_cluster(resized_image)
# #-------------------------------------------------------------#
pupil_ellipse, mean, width, height, angle = find_ellipse(detected_cluster)
origenal_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display
plot_simplefigure(origenal_image, " Eye Frame ", save_directory, "Eye_Frame")
cv2.ellipse(origenal_image, pupil_ellipse, color=(170,86,255), thickness= 2 )
plot_simplefigure(origenal_image, " First fitted ellipse ", save_directory, "Pupil_detection_without_reflection")
detected_cluster_show = cv2.cvtColor(detected_cluster, cv2.COLOR_GRAY2BGR)
plot_simplefigure(detected_cluster_show, "Binarized Pupil Area", save_directory, "Binarized_Pupil_Area")
# #################################Creating second raw Image#######################
origenal_image3 = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
cv2.ellipse(origenal_image3, reflect1_ellipse, color=(170,86,2), thickness= 1 )
cv2.ellipse(origenal_image3, reflect1_ellipse2, color=(170,86,2), thickness= 1 )
plot_simplefigure(origenal_image3, "Reflection", save_directory, "reflection")

detected_cluster2 = cv2.cvtColor(detected_cluster, cv2.COLOR_GRAY2BGR)
cv2.ellipse(detected_cluster2, reflect1_ellipse, color= (255, 255, 255), thickness= -1 )
cv2.ellipse(detected_cluster2, reflect1_ellipse, color=(170,86,2), thickness=1)
pupil_ellipse2, _, _, _, _ = find_ellipse(detected_cluster2)
origenal_image4 = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
cv2.ellipse(origenal_image4, pupil_ellipse2, color=(0,95,191), thickness= 2 )
plot_simplefigure(origenal_image4, "Second fit", save_directory,"second_fit")
#
# #######################################################
# detected_cluster2 = cv2.cvtColor(detected_cluster, cv2.COLOR_GRAY2BGR)
# cv2.ellipse(detected_cluster2, reflect1_ellipse, color= (255, 255, 255), thickness= -1 )
# cv2.ellipse(detected_cluster2, reflect1_ellipse, color=(170,86,2), thickness=1)
# cv2.ellipse(detected_cluster2, reflect1_ellipse2, color= (255, 255, 255), thickness= -1 )
# cv2.ellipse(detected_cluster2, reflect1_ellipse2, color=(170,86,2), thickness= 1 )
# plot_simplefigure(detected_cluster2, "Reflection added to the Binary data",  save_directory, "Binary_reflection")
# pupil_ellipse2, _, _, _, _ = find_ellipse(detected_cluster)
# origenal_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
# cv2.ellipse(origenal_image, pupil_ellipse2, color=(170,86,255), thickness= 1 )
# plot_simplefigure(origenal_image, "2")
# binary_image = Image_binarization(resized_image)
# overlap_reflect_image = overlap_reflect(all_reflects, pupil_ellipse, binary_image)
# overlap_reflect_image = cv2.cvtColor(overlap_reflect_image, cv2.COLOR_GRAY2BGR)
# plt.figure(figsize=(6, 6))
# plt.imshow(overlap_reflect_image)
# plt.title("Ellipse Drawn on Image")
# # plt.axis("off")
# ############################## second fitting #########################
#
# # pupil_ellipse, mean, width, height, angle = find_ellipse(detected_cluster)
# #
# # import numpy as np
# #
# # # Path to the NPZ file
# # npz_file_path = r"C:/Users/faezeh.rabbani/PycharmProjects/FaceProject/test_data/test_images/FaceIt/faceit.npz"
# #
# # # Load the NPZ file
# # data = np.load(npz_file_path, allow_pickle=True)
# #
# # # List all saved variables
# # print("Keys in NPZ file:", list(data.keys()))
# #
# # If video is stored as a NumPy array
# if "video_frames" in data:
#     video_frames = data["video_frames"]  # Shape: (frames, height, width, channels)
#     print(f"Video shape: {video_frames.shape}")  # Check video shape
#
# # If video is stored as binary
# if "video_file" in data:
#     video_bytes = data["video_file"].item()  # Extract binary video
#     video_output_path = "restored_video.mp4"
#
#     # Save binary video to a file
#     with open(video_output_path, "wb") as f:
#         f.write(video_bytes)
#
#     print(f"Video restored and saved as {video_output_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import bottleneck as bn
# file_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs\FaceIt\faceit.npz"
#
# # Load the .npz file
# data = np.load(file_path, allow_pickle=True)
#
# # List the available arrays inside the .npz file
# print(data.files)
# print(data["pupil_dilation"])
# pupil_dilation = data["pupil_dilation"]
# #########
# from sklearn.decomposition import FastICA
# import numpy as np
#
# # Assume pupil_dilation is your time-series trace
# pupil_dilation = np.array(pupil_dilation).reshape(-1, 1)  # Reshape for ICA input
#
# ica = FastICA(n_components=5, random_state=42)  # Adjust components based on data
# independent_components = ica.fit_transform(pupil_dilation)
# print("independent_components", independent_components)
#
# # Plot ICA components
# plt.figure(figsize=(10, 6))
# for i in range(independent_components.shape[1]):
#     plt.subplot(independent_components.shape[1], 1, i + 1)
#     plt.plot(independent_components[:, i], label=f'Component {i+1}')
#     plt.legend()
# plt.suptitle('ICA Components of Pupil Dilation Signal')
# moving_variance = np.diff(independent_components[:, i])
# plt.plot(moving_variance)
# plt.show()
#
# print(independent_components)
# blink_component = independent_components[:, 1]  # Example: Assume 2nd component is blinks
# plt.plot(blink_component, color='r')
# plt.title("Identified Blink Component")
# plt.show()
#
# # Check which component corresponds to blinks (usually large amplitude variations)
#
# ####################
#
#
# moving_variance = bn.move_var(pupil_dilation, window=2, min_count=1)
# normalized_variance = (moving_variance - np.min(moving_variance)) / (np.max(moving_variance) - np.min(moving_variance))
#
#
# moving_variance = np.diff(pupil_dilation)
# normalized_variance = (moving_variance - np.min(moving_variance)) / (np.max(moving_variance) - np.min(moving_variance))
# normalized_pupil_dilation = (pupil_dilation - np.min(pupil_dilation)) / (np.max(pupil_dilation) - np.min(pupil_dilation))
# plt.figure(figsize=(15,2))
# plt.title("dif")
# time = np.arange(1, len(normalized_pupil_dilation))
# plt.plot(time, normalized_pupil_dilation[1:])
# # plt.plot(time, normalized_variance)
# plt.show()
