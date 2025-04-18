import cv2
import os.path
import numpy as np
from FACEIT_codes import pupil_detection
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap
import math

def initialize_attributes(obj, image):
    obj.image_height, obj.image_width =  image.shape
    obj.ratio = 2
    obj.reflect_height = 30
    obj.reflect_width = 30
    obj.Face_frame = None
    obj.Pupil_frame = None
    obj.sub_region = None
    obj.reflect_ellipse = None
    obj.saturation = 0
    obj.contrast = 1
    obj.frame = None
    obj.pupil_ROI = None
    obj.face_ROI = None
    obj.pupil_detection = None
    obj.pupil_ellipse_items = None
    obj.current_ROI = None
    obj.ROI_exist = False
    obj.reflection_center = (obj.image_width // obj.ratio / 2, obj.image_height // obj.ratio / 2)
    obj.oval_center = (obj.image_width // obj.ratio/2, obj.image_height // obj.ratio/2)
    obj.face_rect_center = (obj.image_width // obj.ratio/2, obj.image_height // obj.ratio/2)
    obj.ROI_center = (obj.image_width // obj.ratio/2, obj.image_height // obj.ratio/2)
    obj.Image_loaded = False
    obj.Pupil_ROI_exist = False
    obj.Face_ROI_exist = False
    obj.eye_corner_mode = False
    obj.eyecorner = None
    obj.eye_corner_center = None
    obj.erased_pixels = None
    obj.mnd = 10
    obj.binary_threshold = 220
    obj.Show_biary = False
    obj.clustering_method = "DBSCAN"



def show_ROI(ROI, image):
    sub_image = ROI.rect()
    top = int(sub_image.top())*2
    bottom = int(sub_image.bottom())*2
    left = int(sub_image.left())*2
    right = int(sub_image.right())*2
    sub_region = image[top:bottom, left:right]
    frame = [top,bottom, left,right]
    return sub_region, frame



def change_saturation(image, saturation_scale, contrast, brightness=0):
    """
    Changes the saturation of an image and adjusts brightness to make dark pixels darker and bright pixels brighter.

    Parameters:
        image (numpy.ndarray): The input image in BGR format.
        saturation_scale (float): The scale by which to adjust saturation and brightness.

    Returns:
        numpy.ndarray: The processed image with adjusted saturation and brightness.
    """
    if saturation_scale != 0:

        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * (1 + saturation_scale / 100.0), 0, 255)
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * (1 + (saturation_scale / 100.0)), 0, 255)
        hsv_image = hsv_image.astype(np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    return image

def display_sub_region(graphicsView, sub_region, scene2, ROI, saturation,contrast, mnd, binary_threshold,clustering_method,Show_biary, erased_pixels = None,
                       reflect_ellipse = None, pupil_ellipse_items = None, Detect_pupil = False ):
    if pupil_ellipse_items is not None:
        scene2.removeItem(pupil_ellipse_items)
    for item in scene2.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            scene2.removeItem(item)
            del item
    height, width = sub_region.shape[:2]

    if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
        sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)
    sub_region = change_saturation(sub_region, saturation, contrast)
    sub_region_rgba = cv2.cvtColor(sub_region, cv2.COLOR_BGR2BGRA)

    if Show_biary == True:
        sub_region_binary = pupil_detection.Image_binarization(sub_region, binary_threshold)
        sub_region_binary_rgba = cv2.cvtColor(sub_region_binary, cv2.COLOR_BGR2BGRA)
    else:
        sub_region_binary_rgba = sub_region_rgba

    bytes_per_line = width * 4
    qimage = QtGui.QImage(sub_region_binary_rgba.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    pixmap = QPixmap.fromImage(qimage)
    ############
    scaled_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
    if ROI == "pupil":
        item.setZValue(-1)
    scene2.addItem(item)
    scene2.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

    if Detect_pupil == True:
        pupil_ROI0, P_detected_center, P_detected_width, P_detected_height, angle, _ = pupil_detection.detect_pupil(sub_region_rgba, erased_pixels, reflect_ellipse, mnd, binary_threshold, clustering_method)
        pupil_ellipse_item = QtWidgets.QGraphicsEllipseItem(int(P_detected_center[0] - P_detected_width), int(P_detected_center[1] - P_detected_height),
                                                            P_detected_width*2, P_detected_height*2)

        pupil_ellipse_item.setTransformOriginPoint(int(P_detected_center[0]),
                                                   int(P_detected_center[1]))
        pupil_ellipse_item.setRotation(np.degrees(angle))
        pen = QtGui.QPen(QtGui.QColor("purple"))
        pen.setWidth(1)
        pupil_ellipse_item.setPen(pen)
        scene2.addItem(pupil_ellipse_item)
        pupil_ellipse_items = pupil_ellipse_item


    if graphicsView:
        graphicsView.setScene(scene2)
        graphicsView.setFixedSize(scaled_pixmap.width(), scaled_pixmap.height())
    return pupil_ellipse_items

def second_region(graphicsView_subImage,graphicsView_MainFig,  image_width, image_height):
    scene2 = QtWidgets.QGraphicsScene(graphicsView_subImage)
    graphicsView_subImage.setScene(scene2)
    graphicsView_MainFig.graphicsView_subImage = graphicsView_subImage
    return scene2


def display_region(image, graphicsView_MainFig, image_width, image_height, scene=None):
    if scene is None:
        scene = QtWidgets.QGraphicsScene(graphicsView_MainFig)
    else:
        for item in scene.items():
            if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                scene.removeItem(item)
                del item

    qimage = QtGui.QImage(image.data, image_width, image_height, QtGui.QImage.Format_Grayscale8)
    pixmap = QtGui.QPixmap.fromImage(qimage)

    # Scale the pixmap to fit inside the window
    scaled_pixmap = pixmap.scaled(image_width/2, image_height/2, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
    item.setZValue(-1)
    scene.addItem(item)

    graphicsView_MainFig.setScene(scene)
    scene.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

    # Resize the view
    graphicsView_MainFig.setFixedSize(scaled_pixmap.width(), scaled_pixmap.height())
    graphicsView_MainFig.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    graphicsView_MainFig.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    graphicsView_MainFig.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    return graphicsView_MainFig, scene


def load_npy_by_index(folder_path, index):
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if index < 0 or index >= len(npy_files):
        raise IndexError("Index out of range")
    file_path = os.path.join(folder_path, npy_files[index])
    image = np.load(file_path)
    # original_height, original_width = image.shape
    # aspect_ratio = original_width / original_height
    # image_width = int(image_height * aspect_ratio)
    # image = cv2.resize(image, (image_width, image_height), interpolation = cv2.INTER_AREA)
    return image

def load_frame_by_index(video_path, index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video file {video_path}.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if index < 0 or index >= total_frames:
        raise IndexError("Index out of range")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap.release()
        raise ValueError(f"Error: Could not read frame at index {index}.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # original_height, original_width = frame.shape[:2]
    # aspect_ratio = original_width / original_height
    # image_width = int(image_height * aspect_ratio)
    # resized_frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    cap.release()
    return frame

def setup_sliders(parent,min,max,set_value, orientation):
    Slider = QtWidgets.QSlider(parent)
    if orientation == "vertical":
        Slider.setOrientation(QtCore.Qt.Vertical)
    elif orientation == "horizontal":
        Slider.setOrientation(QtCore.Qt.Horizontal)
    Slider.setMinimum(min)
    Slider.setMaximum(max)
    Slider.setValue(set_value)
    return Slider
def get_stylesheet():
    return """
    QWidget {
        background-color: #3d4242;  /* Light gray background */
        color: #000000;  /* Black text */
    }
    QLabel {
        color: white;  /* White text for all labels */
    }
    
    QCheckBox {
    color: white;  /* White text for all QCheckBox */
    }
    
    QPushButton {
        background-color: #CD853F ;  /* Background for buttons */
        color: white;  /* White text on buttons */
        border: 3px outset #CD853F;
        padding: 4px;
    }
    QPushButton:hover {
        background-color: #c24b23;  /* Darker on hover */
    }
    QLineEdit, QSlider {
        background-color: #3d4242;
        color: #000000;
        border: 1px solid #3d4242;
        padding: 5px;
    }
    """

def set_button_style(widget, widget_type):
    widget.setStyleSheet(f"""
        {widget_type}::groove:horizontal {{
            border: 1px solid #999999;
            height: 8px;
            background: #b0b0b0;
            margin: 2px 0;
        }}
        {widget_type}::handle:horizontal {{
            background: #CD853F;
            border: 1px ridge #CD853F;
            width: 8px;
            height: 20px;
            margin: -7px 0;
            border-radius: 3px;
        }}
    """)

def add_eyecorner(x_pos , y_pos, scene2, graphicsView_subImage):
    if hasattr(graphicsView_subImage, 'eyecorner') and graphicsView_subImage.eyecorner is not None:
        scene2.removeItem(graphicsView_subImage.eyecorner)
    diameter = 2
    eyecorner = QtWidgets.QGraphicsEllipseItem(x_pos-diameter/2 , y_pos-diameter/2, diameter , diameter)
    pen = QtGui.QPen(QtGui.QColor("peru"))
    pen.setWidth(0)
    eyecorner.setPen(pen)
    brush = QtGui.QBrush(QtGui.QColor("peru"))
    eyecorner.setBrush(brush)
    scene2.addItem(eyecorner)
    graphicsView_subImage.eyecorner = eyecorner
    eye_corner_center = (x_pos , y_pos)
    return eye_corner_center




