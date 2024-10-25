import numpy as np
import cv2
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

#------------------------------------------Show all clusters-----------------------------------
#colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# plt.figure(figsize=(8, 6))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         col = [0, 0, 0, 1]
#     class_member_mask = (labels == k)
#     xy = coords[class_member_mask]
#     plt.plot(xy[:, 1], xy[:, 0],'o',  markerfacecolor=tuple(col), markeredgecolor='k', markersize=10)
# plt.gca().invert_yaxis()
# plt.gca().spines['top'].set_position(('data', 0))
# plt.gca().spines['left'].set_position(('data', 0))
# plt.gca().xaxis.set_ticks_position('top')
# plt.gca().yaxis.set_ticks_position('left')
# plt.gca().xaxis.set_label_position('top')
# plt.title(f'Estimated number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}')
# plt.show()
#
# unique_labels, counts = np.unique(labels, return_counts=True)
# biggest_class_label = unique_labels[np.argmax(counts)]
# biggest_class_size = counts[np.argmax(counts)]
