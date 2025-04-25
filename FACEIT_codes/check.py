# import subprocess
#
# input_path = r"C:\Users\faezeh.rabbani\Documents\check\input.mp4"
# output_path = r"C:\Users\faezeh.rabbani\Downloads\output_deinterlaced.mp4"
# ffmpeg_path = r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  # Full path to FFmpeg executable
#
#
# def deinterlace_video(input_path, output_path):
#     """
#       Deinterlace video using FFmpeg's yadif filter.
#       """
#     command = [
#         ffmpeg_path,  # Full path to the ffmpeg executable
#         '-i', input_path,
#         '-vf', 'yadif=1',
#         '-r', '60',
#         output_path
#     ]
#
#     try:
#         subprocess.run(command, check=True)
#         print(f"Deinterlaced video saved to: {output_path}")
#     except subprocess.CalledProcessError as e:
#         print("Error during FFmpeg execution:", e)
#     except FileNotFoundError as e:
#         print(f"Error: {e}. Ensure that FFmpeg is correctly installed and the path is correct.")
#
#
# # Example usage
# deinterlace_video(input_path, output_path)
#
#
#
#
# def display_sub_region(self, sub_region, ROI, Detect_pupil=False):
#     # Remove old items
#     if self.app_instance.pupil_ellipse_items is not None:
#         self.app_instance.scene2.removeItem(self.app_instance.pupil_ellipse_items)
#     for item in self.app_instance.scene2.items():
#         if isinstance(item, QtWidgets.QGraphicsPixmapItem):
#             self.app_instance.scene2.removeItem(item)
#             del item
#
#     # Apply saturation
#     if self.app_instance.saturation_method == "Gradual":
#         if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
#             processed_sub_region = self.apply_intensity_gradient(sub_region)
#             processed_sub_region = cv2.cvtColor(processed_sub_region, cv2.COLOR_GRAY2BGR)
#         else:
#             processed_sub_region = self.change_Gradual_saturation(sub_region)
#             processed_sub_region = cv2.cvtColor(processed_sub_region, cv2.COLOR_RGB2BGR)
#
#     elif self.app_instance.saturation_method == "Uniform":
#
#         processed_sub_region = self.change_saturation_uniform(sub_region)
#     else:
#         processed_sub_region = sub_region.copy()
#
#     # Apply binarization
#     if self.app_instance.Show_binary:
#         binary_image = pupil_detection.Image_binarization(processed_sub_region, self.app_instance.binary_threshold)
#         if len(binary_image.shape) == 2:
#             sub_region_to_present = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGBA)
#         else:
#             sub_region_to_present = cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGBA)
#     else:
#         sub_region_to_present = cv2.cvtColor(processed_sub_region, cv2.COLOR_BGR2RGBA)
#
#     # Create QImage
#     height, width = sub_region_to_present.shape[:2]
#     bytes_per_line = width * 4
#     qimage = QtGui.QImage(sub_region_to_present.data.tobytes(), width, height, bytes_per_line,
#                           QtGui.QImage.Format_RGBA8888)
#     ########################
#
#
#     # Convert to Pixmap and display
#     pixmap = QtGui.QPixmap.fromImage(qimage)
#     scaled_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
#
#     item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
#     if ROI == "pupil":
#         item.setZValue(-1)
#     self.app_instance.scene2.addItem(item)
#     self.app_instance.scene2.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
#
#     # Detect pupil if needed
#     if Detect_pupil:
#         _, P_detected_center, P_detected_width, P_detected_height, angle, _ = pupil_detection.detect_pupil(
#             processed_sub_region, self.app_instance.erased_pixels, self.app_instance.reflect_ellipse,
#             self.app_instance.mnd, self.app_instance.binary_threshold, self.app_instance.clustering_method)
#
#         ellipse_item = QtWidgets.QGraphicsEllipseItem(
#             int(P_detected_center[0] - P_detected_width),
#             int(P_detected_center[1] - P_detected_height),
#             P_detected_width * 2,
#             P_detected_height * 2
#         )
#         ellipse_item.setTransformOriginPoint(int(P_detected_center[0]), int(P_detected_center[1]))
#         ellipse_item.setRotation(np.degrees(angle))
#         ellipse_item.setPen(QtGui.QPen(QtGui.QColor("purple"), 1))
#         self.app_instance.scene2.addItem(ellipse_item)
#         self.app_instance.pupil_ellipse_items = ellipse_item
#
#     if self.app_instance.graphicsView_subImage:
#         self.app_instance.graphicsView_subImage.setScene(self.app_instance.scene2)
#         self.app_instance.graphicsView_subImage.setFixedSize(scaled_pixmap.width(), scaled_pixmap.height())
#
#     return self.app_instance.pupil_ellipse_items
#
#
#
# def detect_pupil(chosen_frame_region, erased_pixels, reflect_ellipse, mnd, binary_threshold, clustering_method):
#     binary_image = Image_binarization(chosen_frame_region,binary_threshold)
#     binary_image = erase_pixels(erased_pixels, binary_image)
#
#     if clustering_method == "DBSCAN":
#         binary_image = find_cluster_DBSCAN(binary_image, mnd)
#     elif clustering_method == "watershed":
#         binary_image, _ = find_cluster_watershed(binary_image)
#     elif clustering_method == "SimpleContour":
#         binary_image = find_cluster_simple(binary_image)
#
#
#     if reflect_ellipse is None or reflect_ellipse == [[], [], []]:
#         pupil_ROI0, center, width, height, angle = find_ellipse(binary_image)
#     else:
#         All_reflects = [
#             [reflect_ellipse[0][variable], (reflect_ellipse[1][variable], reflect_ellipse[2][variable]), 0]
#             for variable in
#             range(len(reflect_ellipse[1]))]
#         for i in range(2):
#             pupil_ROI0, center, width, height, angle = find_ellipse(binary_image)
#             binary_image_update = overlap_reflect(All_reflects, pupil_ROI0, binary_image)
#             binary_image = binary_image_update
#     pupil_area = np.pi * (width*height)
#     return pupil_ROI0, center, width, height, angle, pupil_area
import numpy as np


def get_symmetric_concave_mask(h: int, w: int, direction="Horizontal", strength=2.0):
    size = w if direction in ["Horizontal"] else h
    x = np.linspace(-1, 1, size)
    power = 1.1
    curve = np.abs(x) ** power
    scaled = 1 + (strength - 1) * curve
    if direction == "Horizontal":
        return np.tile(scaled, (h, 1))
    elif direction == "Vertical":
        return np.tile(scaled, (w, 1)).T



h, w = 200, 256
mask = get_symmetric_concave_mask(h, w,  direction="Vertical", strength=2.0)

import matplotlib.pyplot as plt
plt.imshow(mask, cmap="viridis")
plt.title("Convex Gradient Example")
plt.colorbar()
plt.show()
