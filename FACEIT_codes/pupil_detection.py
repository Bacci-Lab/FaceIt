import matplotlib.pyplot as plt
import numpy as np
import cv2
import bottleneck as bn
from sklearn.cluster import DBSCAN

def find_ellipse(binary_image, show=False):
    """
    Fits an ellipse to non-zero pixels in a binary image.

    Parameters:
    binary_image (np.ndarray): A binary image where non-zero pixels are considered for fitting.
    show (bool): If True, plots the fitted ellipse and eigenvectors.

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

    if coords.size == 0:
        return ((0, 0), (0, 0), 0), (0.0, 0.0), 0.0, 0.0, 0.0

    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean

    if len(centered_coords) < 2:
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    cov_matrix = np.cov(centered_coords, rowvar=False)

    if cov_matrix.ndim != 2 or cov_matrix.shape != (2, 2):
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    if np.any(np.isnan(eigenvalues)) or np.any(eigenvalues <= 0):
        return ((0, 0), (0, 0), 0), (float(mean[0]), float(mean[1])), 0.0, 0.0, 0.0

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])

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
    return hull_image



def find_cluster_simple(
    binary_image: np.ndarray,
    show_plot: bool = False,
    *,
    filter_contours: bool = False,
    max_width_frac: float = 0.8,
    max_aspect: float = 2.0,
    min_area: int = 0
) -> np.ndarray:
    """
    Find the main blob and return a convex-hull mask.

    Steps:
      1) find contours
      2) (optional) filter by size/shape
      3) choose largest
      4) draw convex hull of that contour

    Parameters
    ----------
    binary_image : np.ndarray
        2D uint8 mask (0/255).
    show_plot : bool
        If True, show intermediate views.
    filter_contours : bool
        If True, apply width/aspect/area filters; if False, consider all contours.
    max_width_frac : float
        Max allowed width as a fraction of image width (e.g., 0.8 → 80%).
    max_aspect : float
        Max allowed aspect ratio W/H.
    min_area : int
        Minimum contour area to keep (0 disables).

    Returns
    -------
    final_mask : np.ndarray
        Binary mask (uint8) of the convex hull for the chosen contour (or zeros).
    """
    # 1) Contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = binary_image.shape
    kept = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        if not filter_contours:
            kept.append(cnt)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = (w / h) if h > 0 else np.inf
        wide_enough = (w <= max_width_frac * w_img)
        aspect_ok = (aspect <= max_aspect)
        if wide_enough and aspect_ok:
            kept.append(cnt)

    # 3) Largest contour (post-filter or all)
    if kept:
        largest = max(kept, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        final_mask = np.zeros_like(binary_image)
        cv2.drawContours(final_mask, [hull], -1, 255, -1)
    else:
        largest = None
        final_mask = np.zeros_like(binary_image)

    return final_mask




    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     return np.zeros_like(binary_image)
    #
    # image_height, image_width = binary_image.shape
    # filtered_contours = []
    #
    # # Step 1: Remove contours that are too wide
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w <= 0.7 * image_width and (w / h if h > 0 else float('inf')) <= 2:
    #         filtered_contours.append(contour)
    #
    # if not filtered_contours:
    #     print("All contours were too wide — returning empty mask.")
    #     return np.zeros_like(binary_image)
    #
    # # Step 2: Get the largest remaining contour
    # largest = max(filtered_contours, key=cv2.contourArea)
    # hull = cv2.convexHull(largest)
    #
    # # Step 3: Create binary mask
    # mask = np.zeros_like(binary_image)
    # cv2.drawContours(mask, [hull], -1, 255, -1)
    #
    # return mask

def find_cluster_DBSCAN(
    binary_image: np.ndarray,
    mnd: float,
    show_cluster: bool = False,
    *,
    filter_clusters: bool = True,
    max_width_frac: float = 0.8,   # reject clusters wider than this × image width
    max_aspect: float = 2.0,       # reject clusters with W/H > max_aspect
    min_cluster_points: int = 1    # DBSCAN min_samples
) -> np.ndarray:
    """
    Cluster foreground pixels with DBSCAN and return the convex hull of the
    largest valid cluster. When `filter_clusters=False`, all clusters are
    considered valid (no width/aspect filtering).

    Parameters
    ----------
    binary_image : np.ndarray
        2D uint8 mask (0/255).
    mnd : float
        DBSCAN `eps` (larger merges more points).
    show_cluster : bool
        If True, draw the clusters kept after filtering.
    filter_clusters : bool
        If True (default), apply width/aspect filters. If False, skip filtering.
    max_width_frac : float
        Max allowed cluster width as a fraction of image width (only if filtering).
    max_aspect : float
        Max allowed aspect ratio W/H (only if filtering).
    min_cluster_points : int
        DBSCAN `min_samples` (minimum points per cluster).

    Returns
    -------
    hull_image : np.ndarray
        Binary mask of the convex hull of the chosen cluster (or zeros).
    """
    non_zero_coords = np.column_stack(np.where(binary_image > 0))
    if non_zero_coords.shape[0] == 0:
        return np.zeros_like(binary_image)

    image_height, image_width = binary_image.shape

    clustering = DBSCAN(eps=mnd, min_samples=min_cluster_points).fit(non_zero_coords)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    all_clusters_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    filtered_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    valid_clusters = []

    for label in unique_labels:
        if label == -1:  # DBSCAN noise
            continue
        cluster_coords = non_zero_coords[labels == label]
        ys, xs = cluster_coords[:, 0], cluster_coords[:, 1]
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1

        color = np.random.randint(0, 255, size=3).tolist()
        for y, x in cluster_coords:
            all_clusters_img[y, x] = color

        keep = True
        if filter_clusters:
            aspect = (w / h) if h > 0 else float('inf')
            keep = (w <= max_width_frac * image_width) and (aspect <= max_aspect)

        if keep:
            valid_clusters.append(cluster_coords)
            if show_cluster:
                for y, x in cluster_coords:
                    filtered_img[y, x] = color

    if valid_clusters:
        largest_cluster = max(valid_clusters, key=lambda arr: arr.shape[0])
        cluster_mask = np.zeros_like(binary_image)
        cluster_mask[largest_cluster[:, 0], largest_cluster[:, 1]] = 255

        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            hull_image = np.zeros_like(binary_image)
            cv2.drawContours(hull_image, [hull], -1, 255, -1)
        else:
            hull_image = cluster_mask
    else:
        if filter_clusters:
            print("No valid clusters after filtering.")
        hull_image = np.zeros_like(binary_image)

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

def remove_reflection_with_inpaint(gray_image, reflect_ellipses):
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    for center, width, height in zip(*reflect_ellipses):
        center = tuple(map(int, center))
        axes = (int(width // 2), int(height // 2))
        cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    inpainted = cv2.inpaint(gray_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted
def erase_pixels(erased_pixels, binary_image):
    """
    Given a list/array of (x,y) pixels to erase, zero out only those
    that actually fall inside the binary_image bounds.
    """
    if erased_pixels is None or len(erased_pixels) == 0:
        return binary_image

    arr = np.array(erased_pixels, dtype=int)
    # Expect shape (N,2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        h, w = binary_image.shape[:2]
        xs, ys = arr[:, 0], arr[:, 1]
        # keep only points inside [0,w) × [0,h)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        good = arr[valid]
        if good.size > 0:
            binary_image[good[:, 1], good[:, 0]] = 0

    return binary_image


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
def Image_binarization_constant(chosen_frame_region,erased_pixels, binary_threshold = 220, show_binary = False, show_original = False):
    if len(chosen_frame_region.shape) == 3:
        if chosen_frame_region.shape[2] == 4:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGRA2GRAY)
        elif chosen_frame_region.shape[2] == 3:
            sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {chosen_frame_region.shape[2]}")
    else:
        sub_region_2Dgray = chosen_frame_region.copy()
    _, binary_image = cv2.threshold(sub_region_2Dgray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    binary_image = erase_pixels(erased_pixels, binary_image)

    # === Apply Ellipse Mask ===
    height, width = binary_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    if show_binary == True:
        plt.figure(figsize=(10, 8))
        plt.imshow(binary_image, cmap='binary')
        plt.show()


    return binary_image


def detect_reflection_automatically(
    gray_image: np.ndarray,
    bright_thresh: int,
    min_area: int = 10,
    max_area: int = 500,
    circularity_thresh: float = 0.6,
    dilation_factor: float = 4,   # target mask area = 300% (scalable)
    show: bool = True
) -> tuple[np.ndarray, float]:

    # --- Step 1: Compute hybrid threshold ---
    bright_thresh =bright_thresh /10

    thresh_val = np.percentile(gray_image,bright_thresh)
    # === EARLY EXIT IF TOO LOW ===
    if thresh_val < 100:
        return np.zeros_like(gray_image, dtype=np.uint8)


    _, bw = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY)

    # --- Step 2: Find contours ---
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    primary_area = 0.0
    raw_mask = np.zeros_like(gray_image, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 5:
            continue

        # circularity check
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        major, minor = max(MA, ma), min(MA, ma)
        circ = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2) if area > 0 else 0

        if minor > 0 and (minor / major) > circularity_thresh and circ > 0.5:
            primary_area = area
            cv2.drawContours(raw_mask, [cnt], -1, 255, -1)

    # --- Step 3: Dilate mask proportional to area ---
    if primary_area > 0:
        target_area = primary_area * dilation_factor
        radius = np.sqrt(primary_area / np.pi)
        target_radius = np.sqrt(target_area / np.pi)
        extra_radius = int(round(target_radius - radius))
        kernel_size = max(3, 2 * extra_radius + 1)  # odd kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(raw_mask, kernel, iterations=1)
    else:
        dilated_mask = raw_mask.copy()

    # --- Step 4: Visualization ---
    if show:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))

        axes[0, 0].imshow(gray_image, cmap="gray")
        axes[0, 0].set_title("Input Grayscale"); axes[0, 0].axis("off")

        axes[0, 1].imshow(bw, cmap="gray")
        axes[0, 1].set_title(f"Hybrid Threshold > {thresh_val:.1f} (Otsu=)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(raw_mask, cmap="gray")
        axes[0, 2].set_title(f"Raw Mask (area={primary_area:.1f})"); axes[0, 2].axis("off")

        axes[1, 0].imshow(dilated_mask, cmap="gray")
        axes[1, 0].set_title("Dilated Mask"); axes[1, 0].axis("off")

        overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        dil_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, dil_contours, -1, (0, 255, 0), 2)
        axes[1, 1].imshow(overlay[..., ::-1]); axes[1, 1].set_title("Overlay"); axes[1, 1].axis("off")

        comparison = cv2.addWeighted(gray_image, 0.7, dilated_mask, 0.3, 0)
        axes[1, 2].imshow(comparison, cmap="gray")
        axes[1, 2].set_title("Original + Dilated"); axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    return dilated_mask

def Image_binarization_Adaptive(chosen_frame_region, reflect_brightness, c_value,block_size, erased_pixels=None, reflect_ellipse=None, show=False):
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
    if reflect_ellipse is not None and len(reflect_ellipse) > 0:
        sub_region_2Dgray = remove_reflection_with_inpaint(sub_region_2Dgray, reflect_ellipse)

    if reflect_ellipse is None or reflect_ellipse == [[], [], []]:
        refl_mask = detect_reflection_automatically(sub_region_2Dgray, reflect_brightness, show=False)
        if refl_mask.max() > 0:
            sub_region_2Dgray = cv2.inpaint(sub_region_2Dgray, refl_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        sub_region_2Dgray = remove_reflection_with_inpaint(sub_region_2Dgray, reflect_ellipse)
    # sub_region_2Dgray = cv2.medianBlur(sub_region_2Dgray, 7)

    # === Adaptive Thresholding ===
    block_size = ((block_size // 2) * 2) + 1
    binary_image = cv2.adaptiveThreshold(
        sub_region_2Dgray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C= c_value
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
    return binary_image


def detect_pupil(
    chosen_frame_region,
    erased_pixels,
    reflect_ellipse,
    mnd,
    reflect_brightness,
    clustering_method,
    binary_method,
    binary_threshold,
    c_value,
    block_size,
    *,
    disable_filtering: bool = False,   # ← NEW (kw-only, backward compatible)
):

    if binary_method == "Adaptive":
        binary_image = Image_binarization_Adaptive(chosen_frame_region, reflect_brightness,c_value,block_size, erased_pixels=erased_pixels, reflect_ellipse =reflect_ellipse)
    elif binary_method == "Constant":
        binary_image = Image_binarization_constant(chosen_frame_region,erased_pixels, binary_threshold, show_binary = False, show_original = False)

    if clustering_method == "DBSCAN":
        binary_image = find_cluster_DBSCAN(
            binary_image,
            mnd,
            show_cluster=False,
            filter_clusters=not disable_filtering,   # invert checkbox meaning
        )
    elif clustering_method == "watershed":
        result = find_cluster_watershed(binary_image)
        if isinstance(result, tuple):
            binary_image = result[0]
        else:
            binary_image = result
    elif clustering_method == "SimpleContour":
        binary_image = find_cluster_simple(
        binary_image,
        show_plot=False,
        filter_contours=not disable_filtering,
    )
    if binary_method == "Adaptive":
        pupil_ROI0, center, width, height, angle = find_ellipse(binary_image ,show=False)
        pupil_area = np.pi * (width * height)
    elif binary_method == "Constant":
        if reflect_ellipse is None or reflect_ellipse == [[], [], []]:
            pupil_ROI0, center, width, height, angle = find_ellipse(binary_image ,show=False)
        else:
            All_reflects = [
                [reflect_ellipse[0][variable], (reflect_ellipse[1][variable], reflect_ellipse[2][variable]), 0]
                for variable in
                range(len(reflect_ellipse[1]))]
            for i in range(3):
                pupil_ROI0, center, width, height, angle = find_ellipse(binary_image ,show=False)
                binary_image_update = overlap_reflect(All_reflects, pupil_ROI0, binary_image)
                binary_image = binary_image_update


        pupil_area = np.pi * (width * height)
    return pupil_ROI0, center, width, height, angle, pupil_area

