# from pupil_detection import Image_binarization
from sklearn.cluster import DBSCAN
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
save_directory = r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static'
directory_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs"
video_path = r"C:\Users\faezeh.rabbani\Downloads\output_deinterlaced.mp4"
file_index = 11137
video_frame_light_inside = 8708
def load_image(directory_path, file_index):

    files = sorted(os.listdir(directory_path))
    # Check if the specific file index exists
    if file_index < len(files):
        file_name = files[file_index]
        file_path = os.path.join(directory_path, file_name)
        binary_image = np.load(file_path)
    return binary_image

def load_video_frame(video_path,frame_number):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, image_bgr = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number} from video.")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_gray


def plot_simplefigure(image, title, save_directory=None, file_name=None, color=False):
    plt.figure(figsize=(6, 6))
    if color is False:
        plt.imshow(image)
    else:
        if image.ndim == 3:  # If it's RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plt.imshow(image, cmap="binary")
    plt.title(title, fontsize=18)
    plt.axis("off")
    if save_directory:
        save_svg(save_directory, file_name)
    plt.show()


from matplotlib.patches import Ellipse

def find_ellipse(binary_image, labels=None):
    coords = np.column_stack(np.where(binary_image > 0))
    coords = coords[:, [1, 0]]  # Ensure x, y format

    if coords.size == 0:
        return ((0, 0), (0, 0), 0), (0.0, 0.0), 0.0, 0.0, 0.0

    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean
    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])
    ellipse = (int(mean[0]), int(mean[1])), (int(width * 2), int(height * 2)), np.degrees(angle)

    # Call plotting function
    plot_clusters_with_pca(coords, labels if labels is not None else np.zeros(len(coords), dtype=int),
                           mean, eigenvalues, eigenvectors)

    return ellipse, (float(mean[0]), float(mean[1])), width, height, angle


def plot_clusters_with_pca(coords, labels, mean, eigenvalues, eigenvectors):
    from matplotlib.patches import Ellipse

    plt.figure(figsize=(7, 5))
    plt.title("PCA on clustered data with ellipse", fontsize=12)

    # Different colors for different clusters
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', s=4)

    # # Plot eigenvectors
    # for i in range(2):
    #     vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    #     plt.quiver(mean[0], mean[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
    #                color=['teal', 'orange'][i], label=f"PC {i + 1}")
    #
    # # Draw ellipse
    # ellipse_patch = Ellipse(
    #     xy=mean,
    #     width=2 * np.sqrt(eigenvalues[0]) * 2,
    #     height=2 * np.sqrt(eigenvalues[1]) * 2,
    #     angle=np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
    #     edgecolor='black',
    #     facecolor='none',
    #     linestyle='--',
    #     linewidth=1.5,
    #     label='Fitted Ellipse'
    # )
    # plt.gca().add_patch(ellipse_patch)

    # Styling
    plt.legend(frameon=False)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.grid(False)  # <== remove grid
    plt.gca().set_facecolor('none')  # Optional: transparent background
    plt.axis('off')  # Optional: remove axis lines and ticks
    plt.show()



# def find_ellipse(binary_image):
#     """
#     Fits an ellipse to non-zero pixels in a binary image and visualizes eigenvalues and eigenvectors.
#
#     Parameters:
#     binary_image (np.ndarray): A binary image where non-zero pixels are considered for fitting.
#
#     Returns:
#     tuple: A tuple containing:
#         - ellipse ((tuple), (tuple), float): Center (x, y), axes lengths (width, height), and rotation angle in degrees.
#         - mean (tuple): Center of the fitted ellipse (x, y).
#         - width (float): Width of the fitted ellipse (major axis length).
#         - height (float): Height of the fitted ellipse (minor axis length).
#         - angle (float): Rotation angle of the fitted ellipse in radians.
#     """
#     # Get coordinates of non-zero pixels
#     coords = np.column_stack(np.where(binary_image > 0))
#     coords = coords[:, [1, 0]]  # Ensure x, y format
#
#     # Return default values if no coordinates are found
#     if coords.size == 0:
#         return ((0, 0), (0, 0), 0), (0.0, 0.0), 0.0, 0.0, 0.0
#
#     # Calculate mean position and center the coordinates
#     mean = np.mean(coords, axis=0)
#     centered_coords = coords - mean
#
#     # Compute covariance matrix and its eigenvalues and eigenvectors
#     cov_matrix = np.cov(centered_coords, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#
#     # Sort eigenvalues and eigenvectors in descending order
#     sorted_indices = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[sorted_indices]
#     eigenvectors = eigenvectors[:, sorted_indices]
#
#     # Calculate the ellipse properties
#     angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
#     width = 2 * np.sqrt(eigenvalues[0])
#     height = 2 * np.sqrt(eigenvalues[1])
#
#     # Create the ellipse representation
#     ellipse = (int(mean[0]), int(mean[1])), (int(width * 2), int(height * 2)), np.degrees(angle)
#
#     # Visualization
#     plt.figure(figsize=(7, 5))
#     plt.title("Application of PCA on pupil data", fontsize = 12)
#     # Plot data points
#     plt.scatter(coords[:, 0], coords[:, 1], color='mediumorchid', s=4, label="pupil pixels")
#
#     # Plot eigenvectors scaled by eigenvalues
#     for i in range(2):
#         vector = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2  # Scale by eigenvalues
#         plt.quiver(mean[0], mean[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1,
#                    color=['darkcyan', 'orange'][i], label=f"PC {i+1} ")
#
#     # Add legend and axis labels
#     plt.legend(frameon=False)
#
#     # Optionally, remove spines and ticks for a clean look
#     ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.axis('equal')  # Ensure equal scaling
#     plt.grid(True)
#     plt.gca().invert_yaxis()
#     # plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\PCA_application.svg',
#     #             format='svg', transparent=True)
#     plt.show()
#     plt.close()
#     return ellipse, (float(mean[0]), float(mean[1])), width, height, angle
def Image_binarization(chosen_frame_region):
    # Check if the image is already grayscale
    if len(chosen_frame_region.shape) == 2:  # single channel, already grayscale
        sub_region_2Dgray = chosen_frame_region
    else:
        # Convert to grayscale if it has multiple channels (e.g., RGB)
        sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(sub_region_2Dgray, 230, 255, cv2.THRESH_BINARY_INV)
    return binary_image


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
    ax2.set_facecolor('none')  # Set axis background to transparent
    plt.title("Image Clustered")
    plt.scatter(non_zero_coords[:, 1], non_zero_coords[:, 0], c=labels, cmap='tab20b', s=1)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\docs\build\html\_static\Image_clustered.png',
                 transparent=True)
    plt.close()
def save_svg(save_directory_path, file_name):
    file_name = file_name + ".svg"
    file_path = os.path.join(save_directory_path, file_name)
    plt.savefig(file_path,
                format='svg', transparent=True)
from matplotlib.patches import Ellipse
def remove_reflection_with_inpaint(gray_image, reflect_ellipses):
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    for center, width, height in zip(*reflect_ellipses):
        center = tuple(map(int, center))
        axes = (int(width // 2), int(height // 2))
        cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    inpainted = cv2.inpaint(gray_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted
def detect_reflection_automatically(
    gray_image: np.ndarray,
    bright_thresh: int ,
    min_area: int = 10,
    max_area: int = 500,
    circularity_thresh: float = 0.6,
    dilation_radius: int = 5,
    show: bool = False
) -> tuple[np.ndarray, float]:
    """
    Automatically find small bright, roughly circular spots (reflections)
    in a grayscale image and return a binary mask of them (dilated)
    *and* the area of the chosen (primary) contour.
    If show=True, plots the input, raw threshold, and final mask.
    """
    # 1) Threshold the brightest pixels
    _, bw = cv2.threshold(gray_image, bright_thresh, 255, cv2.THRESH_BINARY)

    # 2) Find contours on that threshold
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    primary_area = 0.0
    raw_mask = np.zeros_like(gray_image, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # skip tiny sets of points
        if len(cnt) < 5:
            continue

        # check ellipse circularity
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        major, minor = max(MA, ma), min(MA, ma)
        circ = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2) if area > 0 else 0

        if minor > 0 and (minor / major) > circularity_thresh and circ > 0.5:
            # this is our primary reflection contour:
            primary_area = area
            cv2.drawContours(raw_mask, [cnt], -1, 255, -1)
            # break  # stop after first valid one

    # 3) Dilate that region
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )
    dilated_mask = cv2.dilate(raw_mask, kernel, iterations=1)

    if show:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0,0].imshow(gray_image, cmap='gray')
        axes[0,0].set_title("Input Grayscale")
        axes[0,0].axis('off')

        axes[0,1].imshow(bw, cmap='gray')
        axes[0,1].set_title(f"Threshold > {bright_thresh}")
        axes[0,1].axis('off')

        axes[1,0].imshow(raw_mask, cmap='gray')
        axes[1,0].set_title(f"Raw Mask (area={primary_area:.1f})")
        axes[1,0].axis('off')

        axes[1,1].imshow(dilated_mask, cmap='gray')
        axes[1,1].set_title(f"Dilated Mask (r={dilation_radius})")
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()
    return dilated_mask
def Image_binarization_adaptive(chosen_frame_region, reflect_brightness= 230, erased_pixels=None, reflect_ellipse=None, show=False):
    """
    Applies adaptive binary thresholding. Supports erasing specific pixels or known reflections.
    Also applies an elliptical mask to focus only on the pupil region.
    """
    if len(chosen_frame_region.shape) == 3:
        if chosen_frame_region.shape[2] == 4:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGRA2GRAY)
        elif chosen_frame_region.shape[2] == 3:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {chosen_frame_region.shape[2]}")
    else:
        sub_region_2Dgray = chosen_frame_region.copy()

    # === Normalize non-erased pixels ===
    valid_pixels = sub_region_2Dgray[sub_region_2Dgray > 0]
    if valid_pixels.size > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            sub_region_2Dgray = (sub_region_2Dgray.astype(np.float32) - min_val) / (max_val - min_val) * 255
        else:
            sub_region_2Dgray = np.zeros_like(sub_region_2Dgray, dtype=np.float32)
    else:
        sub_region_2Dgray = np.zeros_like(sub_region_2Dgray, dtype=np.float32)

    sub_region_2Dgray = np.clip(sub_region_2Dgray, 0, 255).astype(np.uint8)

    # fig1, ax1 = plt.subplots()
    # ax1.imshow(sub_region_2Dgray, cmap='gray')
    # ax1.add_patch(plt.Rectangle((60 - (17 // 2), 80 - (17 // 2)), 17, 17, linewidth=2, edgecolor='red', facecolor='none'))
    # ax1.set_title("Neighborhood Location in Image")
    # ax1.axis('off')

    # if reflect_ellipse is not None and len(reflect_ellipse) > 0:
    #     sub_region_2Dgray = remove_reflection_with_inpaint(sub_region_2Dgray, reflect_ellipse)

    # sub_region_2Dgray_with_reflect = sub_region_2Dgray.copy()
    # if reflect_ellipse is None or reflect_ellipse == [[], [], []]:
    #     refl_mask = detect_reflection_automatically(sub_region_2Dgray, reflect_brightness, show=False)
    #     if refl_mask.max() > 0:
    #         sub_region_2Dgray = cv2.inpaint(sub_region_2Dgray, refl_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # else:
    #     sub_region_2Dgray = remove_reflection_with_inpaint(sub_region_2Dgray, reflect_ellipse)
    #
    # sub_region_2Dgray = cv2.medianBlur(sub_region_2Dgray, 7)

    # === Show 17×17 Neighborhood context and thresholding details ===
    center_x, center_y = 140, 80
    half_block = 17 // 2
    neighborhood = sub_region_2Dgray[center_y - half_block:center_y + half_block + 1,
                                     center_x - half_block:center_x + half_block + 1]
    local_mean = np.mean(neighborhood)
    C = 5
    threshold_val = local_mean - C
    center_pixel_val = sub_region_2Dgray[center_y, center_x]

    # # Plot 1: Whole grayscale image with neighborhood box
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(sub_region_2Dgray, cmap='gray')
    # ax1.add_patch(plt.Rectangle((center_x - half_block, center_y - half_block), 17, 17, linewidth=3, edgecolor='red', facecolor='none'))
    # ax1.set_title("Neighborhood Location in Image")
    # ax1.axis('off')
    #
    # # Plot 2: Neighborhood with center pixel highlighted
    # fig2, ax2 = plt.subplots()
    # ax2.imshow(neighborhood, cmap='gray')
    # ax2.add_patch(plt.Rectangle((half_block - 0.5, half_block - 0.5), 1, 1, edgecolor='red', facecolor='none', linewidth=3))
    # ax2.set_title("17×17 Neighborhood with Center Pixel Highlighted")
    # ax2.axis('off')
    #
    # # Plot 3: Histogram of pixel intensities
    # fig3, ax3 = plt.subplots()
    # ax3.hist(neighborhood.flatten(), bins=20, color='gray', linewidth=3)
    # ax3.axvline(local_mean, color='slateblue', linewidth=3, label=f"Mean: {local_mean:.1f}")
    # ax3.axvline(threshold_val, color='teal', linestyle='--', linewidth=3, label=f"Thresh (Mean - C): {threshold_val:.1f}")
    # ax3.axvline(center_pixel_val, color='red', linestyle=':', linewidth=3, label=f"Center Pixel: {center_pixel_val}")
    # ax3.set_xlabel("Pixel Intensity")
    # ax3.set_ylabel("Number of Pixels")
    # ax3.set_title("Pixel Intensities in 17×17 Neighborhood")
    # ax3.legend()
    #
    # # Plot 4: Final decision text
    # fig4, ax4 = plt.subplots()
    # ax4.text(0.1, 0.5, f"Center pixel value: {center_pixel_val}\nThreshold: {threshold_val:.1f}\n\n"
    #                    f"Result: {'WHITE' if center_pixel_val < threshold_val else 'BLACK'}",
    #          fontsize=14, verticalalignment='top')
    # ax4.set_title("Thresholding Decision")
    # ax4.axis('off')
    #
    # plt.show()

    # === Adaptive Thresholding ===
    binary_image = cv2.adaptiveThreshold(
        sub_region_2Dgray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=17,
        C=5
    )

    # === Apply Ellipse Mask ===
    height, width = binary_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Close small gaps: dilate → erode
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Erase specific pixels
    if erased_pixels is not None and len(erased_pixels) > 0:
        if not isinstance(erased_pixels, np.ndarray):
            erased_pixels = np.array(erased_pixels)
        if erased_pixels.ndim == 2 and erased_pixels.shape[1] == 2:
            arr = np.array(erased_pixels, dtype=int)
            h, w = binary_image.shape[:2]
            xs, ys = arr[:, 0], arr[:, 1]
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            good = arr[valid]
            binary_image[good[:, 1], good[:, 0]] = 0

    # Final display
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].imshow(sub_region_2Dgray, cmap='gray')
    axes[0].set_title('Interpolated Light Reflection')
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title('Binary Output after Adaptive Thresholding')
    plt.tight_layout()
    plt.show()

    return binary_image

def find_cluster(resized_image, binary_type="global"):
    """
    Detects clusters of non-zero pixels in a binary image and returns:
    - a binary image showing the convex hull of the largest valid cluster,
    - a color image of all clusters (each in a different color),
    - and a PCA visualization.

    Parameters:
    resized_image (np.ndarray): Input image to be binarized and processed.

    Returns:
    tuple:
        - hull_image (np.ndarray): Binary image with convex hull of largest valid cluster.
        - all_clusters_image (np.ndarray): Color image with all clusters.
        - largest_cluster_mask (np.ndarray): Binary mask of the largest valid cluster.
    """
    # Step 1: Binarize
    if binary_type == "global":
        binary_image = Image_binarization(resized_image)
    elif binary_type == "adaptive":
        binary_image = Image_binarization_adaptive(resized_image)
    else:
        raise ValueError("binary_type must be 'global' or 'adaptive'.")

    non_zero_coords = np.column_stack(np.where(binary_image > 0))  # (y, x)
    if non_zero_coords.shape[0] == 0:
        empty = np.zeros_like(binary_image, dtype=np.uint8)
        return empty, empty, empty

    # Step 2: Clustering
    clustering = DBSCAN(eps=2, min_samples=1).fit(non_zero_coords)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    image_height, image_width = binary_image.shape
    valid_clusters = []
    cluster_masks = []

    # Step 3: Filter valid clusters
    for label in unique_labels:
        cluster_coords = non_zero_coords[labels == label]
        ys, xs = cluster_coords[:, 0], cluster_coords[:, 1]
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1

        if w <= 0.8 * image_width and (w / h if h > 0 else float('inf')) <= 2:
            valid_clusters.append(cluster_coords)

    if not valid_clusters:
        print("No valid clusters after filtering.")
        empty = np.zeros_like(binary_image, dtype=np.uint8)
        return empty, empty, empty

    # Step 4: Select largest valid cluster
    largest_cluster = max(valid_clusters, key=lambda arr: arr.shape[0])
    largest_cluster_mask = np.zeros_like(binary_image, dtype=np.uint8)
    for y, x in largest_cluster:
        largest_cluster_mask[y, x] = 255

    # Step 5: Compute convex hull of the largest valid cluster
    contours, _ = cv2.findContours(largest_cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        hull_image = np.zeros_like(binary_image)
        cv2.drawContours(hull_image, [hull], -1, 255, -1)
    else:
        hull_image = largest_cluster_mask

    # Step 6: Generate color image for all clusters
    all_clusters_image = np.zeros((*binary_image.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    color_map = {label: tuple(int(c) for c in rng.integers(0, 255, size=3)) for label in unique_labels}
    for (y, x), label in zip(non_zero_coords, labels):
        color = color_map[label]
        all_clusters_image[y, x] = color

    # Step 7: Optional PCA plot
    coords_xy = non_zero_coords[:, [1, 0]]
    mean = np.mean(coords_xy, axis=0)
    centered = coords_xy - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    plot_clusters_with_pca(coords_xy, labels, mean, eigenvalues[::-1], eigenvectors[:, ::-1])

    return hull_image, all_clusters_image, largest_cluster_mask



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
def define_ellipse_parameters2():
    # Define the ellipse parameters
    center = (210, 305)  # (x, y)
    axes = (115, 85)  # (width/2, height/2)
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
        cv2.ellipse(pupil_mask, pupil_ellipse, color=255, thickness=-1)

        # Draw each reflection ellipse on the reflection mask
        for reflection in reflections:
            cv2.ellipse(reflection_mask, reflection, color=255, thickness=-1)

        # Identify overlapping regions between the pupil and reflection masks
        overlap_mask = cv2.bitwise_and(pupil_mask, reflection_mask)

        # Find the coordinates of the overlapping pixels
        overlap_coords = np.column_stack(np.where(overlap_mask > 0))

        # Update the binary image at the overlapping coordinates
        binary_image[overlap_coords[:, 0], overlap_coords[:, 1]] = 255

    # Return the modified binary image
    return binary_image

########################################################
reflect1_ellipse = ((93, 63), (30, 27), np.float64(86.59735845915252))
reflect1_ellipse2 = ((87, 84), (15, 15), np.float64(86.59735845915252))
all_reflects = [reflect1_ellipse, reflect1_ellipse2]
# binary_image = load_image(directory_path, file_index)
binary_image = load_video_frame(video_path, video_frame_light_inside)
center, axes, angle = define_ellipse_parameters2()
Image_height, Image_width = binary_image.shape
cropped_image, resized_image = crop_Image(binary_image, center,angle, axes)
resized_image = cropped_image

# detected_cluster, all_clusters_image = find_cluster(resized_image)
_, all_clusters_image, detected_cluster = find_cluster(resized_image, binary_type="adaptive")

#-------------------------------------------------------------#
pupil_ellipse, mean, width, height, angle = find_ellipse(detected_cluster)
origenal_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
plot_simplefigure(origenal_image, " Eye Frame ", save_directory, "Eye_Frame")
cv2.ellipse(origenal_image, pupil_ellipse, color=(170,86,255), thickness= 2 )
plot_simplefigure(origenal_image, " First fitted ellipse ", save_directory, "Pupil_detection_without_reflection")
plot_simplefigure(detected_cluster, "Binarized Pupil Area", save_directory, "Binarized_Pupil_Area", color=True)
#################################Creating second raw Image#######################
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



#######################################################
detected_cluster2 = cv2.cvtColor(detected_cluster, cv2.COLOR_GRAY2BGR)
plot_simplefigure(detected_cluster2, "Reflection added to the Binary data",  save_directory, "Binary_reflection")
pupil_ellipse2, _, _, _, _ = find_ellipse(detected_cluster)
origenal_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
cv2.ellipse(origenal_image, pupil_ellipse2, color=(170,86,255), thickness= 1 )
plot_simplefigure(origenal_image, "2")

binary_image = Image_binarization(resized_image)
overlap_reflect_image = overlap_reflect(all_reflects, pupil_ellipse, binary_image)
overlap_reflect_image = cv2.cvtColor(overlap_reflect_image, cv2.COLOR_GRAY2BGR)
plt.figure(figsize=(6, 6))
plt.imshow(overlap_reflect_image)
plt.title("Ellipse Drawn on Image")
# plt.axis("off")
############################## second fitting #########################
pupil_ellipse, mean, width, height, angle = find_ellipse(detected_cluster)