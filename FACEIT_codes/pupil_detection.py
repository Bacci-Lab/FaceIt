import numpy as np
import cv2
import bottleneck as bn
from sklearn.cluster import DBSCAN
def find_ellipse(binary_image):
    coords = np.column_stack(np.where(binary_image > 0))
    coords = coords[:, [1, 0]]
    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean
    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    if np.isnan(eigenvalues[0]) :
        print("this is true", eigenvalues[0])
        ellipse = (0, 0), (0,0), 0
        mean = (float(0), float(0))
        width = 0
        height= 0
        angle = float(0)
        return ellipse, mean, width, height, angle
    else:
        # Sort the eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Calculate the angle of the ellipse
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        # Calculate the width and height of the ellipse (2 standard deviations)
        width = 2 * np.sqrt(eigenvalues[0])
        height = 2 * np.sqrt(eigenvalues[1])

        # Create the ellipse
        ellipse = (int(mean[0]), int(mean[1])), (int(width * 2), int(height * 2)), np.degrees(angle)
        return ellipse, mean, width, height, angle

def overlap_reflect(reflects, pupil_ellipse, binary_image):
    if reflects != None:
        mask_pupil = np.zeros(binary_image.shape, dtype = np.uint8)
        mask_reflect = np.zeros(binary_image.shape, dtype = np.uint8)
        cv2.ellipse(mask_pupil, pupil_ellipse, 255, -1)
        for i in range(len(reflects)):
            cv2.ellipse(mask_reflect, reflects[i], 255, -1)
        common_mask = cv2.bitwise_and(mask_pupil, mask_reflect)
        coords_common = np.column_stack(np.where(common_mask > 0))
        binary_image[coords_common[:, 0], coords_common[:, 1]] = 255
    return binary_image



def find_claster(binary_image):
    coords = np.column_stack(np.where(binary_image > 0))
    if coords.shape[0] == 0:
        detected_cluster = np.zeros(binary_image.shape, dtype=np.uint8)
    else:
        db = DBSCAN(eps=6, min_samples=1).fit(coords)
        labels = db.labels_
        unique_labels, counts = np.unique(labels, return_counts = True)
        biggest_class_label = unique_labels[np.argmax(counts)]
        class_member_mask = (labels == biggest_class_label)
        xy = coords[class_member_mask]
        detected_cluster = np.zeros(binary_image.shape, dtype=np.uint8)
        for point in xy:
            cv2.circle(detected_cluster,(point[1], point[0]), 1, (255,), -1)
    return detected_cluster
def detect_blinking_ids(pupil,thr, window=8):
    blink_detection = bn.move_var(pupil, window=window, min_count=1)
    threshold = (np.max(blink_detection) - np.min(blink_detection)) /thr
    blink_indices = {i for i, val in enumerate(blink_detection) if val > threshold}
    blink_indices.update({i + j for i in blink_indices for j in range(-1, 2)})
    blink_ids_sorted = sorted(blink_indices)
    return blink_ids_sorted
def interpolate(blinking_id,data):
    mask = np.ones(len(data), dtype=bool)
    mask[blinking_id] = False
    x_full = np.arange(len(data))
    x_valid = x_full[mask]
    data_valid = data[mask]
    data_interpolated = np.interp(x_full, x_valid, data_valid)
    return data_interpolated
