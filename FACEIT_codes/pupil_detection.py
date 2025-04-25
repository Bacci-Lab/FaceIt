import matplotlib.pyplot as plt
import numpy as np
import cv2
import bottleneck as bn
from sklearn.cluster import DBSCAN
from line_profiler import profile
def find_ellipse(binary_image):
    """
    Fits an ellipse to non-zero pixels in a binary image.

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
    coords = coords[:, [1, 0]]  # Ensure x, y format

    # Return default values if no coordinates are found
    if coords.size == 0:
        return ((0, 0), (0, 0), 0), (0.0, 0.0), 0.0, 0.0, 0.0

    # Calculate mean position and center the coordinates
    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean

    # Compute covariance matrix and check its validity
    if len(centered_coords) < 2:
        # Not enough points to calculate covariance matrix
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    cov_matrix = np.cov(centered_coords, rowvar=False)

    # Ensure covariance matrix is 2D
    if cov_matrix.ndim != 2 or cov_matrix.shape != (2, 2):
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Handle potential issues with eigenvalues
    if np.any(np.isnan(eigenvalues)) or np.any(eigenvalues <= 0):
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the ellipse properties
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])

    # Create the ellipse representation
    ellipse = (int(mean[0]), int(mean[1])), (int(width * 2), int(height * 2)), np.degrees(angle)
    return ellipse, (float(mean[0]), float(mean[1])), width, height, angle


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


        overlap_mask = cv2.bitwise_and(pupil_mask, reflection_mask)

        # Find the coordinates of the overlapping pixels
        overlap_coords = np.column_stack(np.where(overlap_mask > 0))

        # Update the binary image at the overlapping coordinates
        binary_image[overlap_coords[:, 0], overlap_coords[:, 1]] = 255

    # Return the modified binary image
    return binary_image

def find_cluster_watershed(binary):
    """
    Applies watershed segmentation to a binary mask to separate touching blobs,
    plots all intermediate results, and returns the final convex hull mask.

    Parameters:
    - binary_mask (np.ndarray): Binary image (uint8) of a single connected region.
    - mnd: (Unused; kept for compatibility with interface)

    Returns:
    - hull_image (np.ndarray): Binary mask (uint8) of the convex hull around all separated regions.
    """

    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Sure foreground (peaks of objects)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Ensure background is not 0
    markers[unknown == 255] = 0  # Unknown region

    # Apply watershed
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)

    # === Plot 1: Watershed-separated clusters in color ===
    color_img = np.zeros((*binary.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(markers)
    rng = np.random.default_rng(42)
    for label in unique_labels:
        if label <= 1:
            continue  # skip background and border
        color = rng.integers(0, 255, 3).tolist()
        color_img[markers == label] = color

    # Remove watershed borders (-1) and prepare binary mask
    markers[markers == -1] = 0
    label_mask = np.where(markers > 1, 255, 0).astype(np.uint8)
    # === Convex Hull computation ===
    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary)

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    hull_image = np.zeros_like(binary)
    cv2.drawContours(hull_image, [hull], -1, 255, -1)
    return hull_image, color_img


def find_cluster_simple(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary_image)

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)  # <-- Apply convexHull here

    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask


def find_cluster_DBSCAN(binary_image, mnd):
    """
    Detects the largest cluster of non-zero pixels in a binary image and returns an image
    highlighting the detected cluster.

    Parameters:
    binary_image (np.ndarray): A binary image with non-zero pixels representing objects.
    mnd (float): Maximum neighbor distance for DBSCAN.

    Returns:
    np.ndarray: A binary image with the largest cluster highlighted.
    """
    # Get coordinates of non-zero pixels in the binary image
    non_zero_coords = np.column_stack(np.where(binary_image > 0))

    # If no non-zero pixels are found, return an empty image
    if non_zero_coords.shape[0] == 0:
        return np.zeros_like(binary_image, dtype=np.uint8)

    # Apply the DBSCAN clustering algorithm to the coordinates
    clustering = DBSCAN(eps=mnd, min_samples=1).fit(non_zero_coords)
    labels = clustering.labels_

    # Find the label of the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_coords = non_zero_coords[labels == largest_cluster_label]

    # Create an image with the largest cluster
    cluster_image = np.zeros_like(binary_image, dtype=np.uint8)
    for y, x in largest_cluster_coords:
        cluster_image[y, x] = 255

    # Find contours
    contours, _ = cv2.findContours(cluster_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cluster_image  # No contours found

    # Merge all contour points
    all_points = np.vstack(contours)

    # Compute convex hull
    hull = cv2.convexHull(all_points)

    # Draw the convex hull
    hull_image = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.drawContours(hull_image, [hull], -1, 255, -1)
    return hull_image


def detect_blinking_ids(pupil_data, threshold_factor, window_size=8):
    """
    Detects blinking indices in pupil data based on a moving variance threshold.

    Parameters:
    pupil_data (np.ndarray): The array of pupil size data.
    threshold_factor (float): The factor used to calculate the variance threshold.
    window_size (int, optional): The size of the moving window for variance calculation. Default is 8.

    Returns:
    list: A sorted list of indices where blinking is detected.
    """
    # Calculate the moving variance of the pupil data
    moving_variance = bn.move_var(pupil_data, window=window_size, min_count=1)

    # Calculate the threshold for blinking detection
    variance_threshold = (np.max(moving_variance) - np.min(moving_variance)) / threshold_factor

    # Identify the indices where the moving variance exceeds the threshold
    detected_blink_indices = {index for index, value in enumerate(moving_variance) if value > variance_threshold}

    # Extend the detected indices by including neighboring indices (-1 to +1)
    expanded_blink_indices = detected_blink_indices.union(
        {index + offset for index in detected_blink_indices for offset in range(-1, 2)}
    )
    # Return a sorted list of unique blink indices
    return sorted(expanded_blink_indices)


def interpolate(blink_indices, data_series):
    """
    Interpolates missing or invalid data points in a data series at specified indices.

    Parameters:
    blink_indices (list or np.ndarray): Indices in the data_series that need interpolation.
    data_series (np.ndarray): The original data series containing valid and invalid data points.

    Returns:
    np.ndarray: A data series with interpolated values at the specified indices.
    """
    # Create a mask array where True indicates valid data and False indicates indices to be interpolated
    valid_mask = np.ones(len(data_series), dtype=bool)
    valid_mask[blink_indices] = False  # Mark indices of blinking as False (invalid)

    # Create an array of indices for the full length of the data series
    all_indices = np.arange(len(data_series))

    # Extract the indices and data values for valid data points
    valid_indices = all_indices[valid_mask]
    valid_data = data_series[valid_mask]

    # Perform interpolation for the full index range using valid data points
    interpolated_data_series = np.interp(all_indices, valid_indices, valid_data)

    # Return the fully interpolated data series
    return interpolated_data_series

def Image_binarization(chosen_frame_region, binary_threshold=220, erased_pixels=None):
    """
    Applies binary thresholding to a normalized grayscale, BGR, or BGRA image.
    Supports erasing specific pixels before thresholding.
    """
    if len(chosen_frame_region.shape) == 3:
        if chosen_frame_region.shape[2] == 4:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGRA2GRAY)
        elif chosen_frame_region.shape[2] == 3:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {chosen_frame_region.shape[2]}")
    else:
        sub_region_2Dgray = chosen_frame_region


    # Erase pixels BEFORE thresholding
    if erased_pixels is not None and len(erased_pixels) > 0:
        if not isinstance(erased_pixels, np.ndarray):
            erased_pixels = np.array(erased_pixels)
        if erased_pixels.ndim == 2 and erased_pixels.shape[1] == 2:
            sub_region_2Dgray[erased_pixels[:, 1], erased_pixels[:, 0]] = 255
    # Normalize intensities to [0, 255]
    sub_region_2Dgray = cv2.normalize(sub_region_2Dgray, None, 0, 255, cv2.NORM_MINMAX)
    # _, binary_image = cv2.threshold(sub_region_2Dgray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.adaptiveThreshold(
        sub_region_2Dgray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,  # or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        cv2.THRESH_BINARY_INV,
        blockSize=25,  # adjust based on pupil size
        C=5  # offset (higher value = darker threshold)
    )

    return binary_image


# def Image_binarization(chosen_frame_region, binary_threshold=220):
#     """
#     Applies binary thresholding to a grayscale, BGR, or BGRA image.
#
#     Parameters:
#         chosen_frame_region (np.ndarray): Input image (grayscale, BGR, or BGRA).
#         binary_threshold (int): Threshold value for binarization.
#
#     Returns:
#         np.ndarray: Binary (single-channel) image.
#     """
#     if len(chosen_frame_region.shape) == 3:
#         if chosen_frame_region.shape[2] == 4:
#             # Handle BGRA images
#             sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGRA2GRAY)
#         elif chosen_frame_region.shape[2] == 3:
#             # Handle BGR images
#             sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)
#         else:
#             raise ValueError(f"Unsupported number of channels: {chosen_frame_region.shape[2]}")
#     else:
#         # Already grayscale
#         sub_region_2Dgray = chosen_frame_region
#
#     _, binary_image = cv2.threshold(sub_region_2Dgray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
#     return binary_image



def detect_pupil(chosen_frame_region, erased_pixels, reflect_ellipse, mnd, binary_threshold, clustering_method):

    binary_image = Image_binarization(chosen_frame_region, binary_threshold, erased_pixels=erased_pixels)

    if clustering_method == "DBSCAN":
        binary_image = find_cluster_DBSCAN(binary_image, mnd)
    elif clustering_method == "watershed":
        result = find_cluster_watershed(binary_image)
        if isinstance(result, tuple):
            binary_image = result[0]
        else:
            binary_image = result
    elif clustering_method == "SimpleContour":
        binary_image = find_cluster_simple(binary_image)

    if reflect_ellipse is None or reflect_ellipse == [[], [], []]:
        pupil_ROI0, center, width, height, angle = find_ellipse(binary_image)
    else:
        All_reflects = [
            [reflect_ellipse[0][variable], (reflect_ellipse[1][variable], reflect_ellipse[2][variable]), 0]
            for variable in range(len(reflect_ellipse[1]))
        ]
        for _ in range(3):
            pupil_ROI0, center, width, height, angle = find_ellipse(binary_image)
            binary_image = overlap_reflect(All_reflects, pupil_ROI0, binary_image)

    pupil_area = np.pi * (width * height)
    return pupil_ROI0, center, width, height, angle, pupil_area



# def erase_pixels(erased_pixels, binary_image):
#     if erased_pixels is not None and len(erased_pixels) > 0:
#         if not isinstance(erased_pixels, np.ndarray):
#             erased_pixels = np.array(erased_pixels)
#         # Ensure the array is of shape (N, 2) for valid indexing
#         if erased_pixels.ndim == 2 and erased_pixels.shape[1] == 2:
#             # Set those pixels to 0 in the binary image
#             binary_image[erased_pixels[:, 1], erased_pixels[:, 0]] = 0
#     return binary_image
