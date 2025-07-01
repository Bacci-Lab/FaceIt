import cv2
import os.path
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
def initialize_attributes(obj, image):
    if len(image.shape) == 3:
        obj.image_height, obj.image_width, _ = image.shape
    elif len(image.shape) == 2:
        obj.image_height, obj.image_width = image.shape
    obj.ratio = 2
    obj.reflect_height = 30
    obj.reflect_width = 30
    obj.Face_frame = None
    obj.Pupil_frame = None
    obj.sub_region = None
    obj.reflect_ellipse = None
    obj.saturation = 0
    obj.saturation_ununiform = 1
    obj.contrast = 1
    obj.brightness = 1
    obj.secondary_BrightGain = 1
    obj.brightness_curve = 1
    obj.frame = None
    obj.pupil_ROI = None
    obj.face_ROI = None
    obj.pupil_detection = None
    obj.pupil_ellipse_items = None
    obj.current_ROI = None
    obj.ROI_exist = False
    obj.reflection_center = (obj.image_width // obj.ratio / 2, obj.image_height // obj.ratio / 2)
    obj.oval_center = (obj.image_width // obj.ratio / 2, obj.image_height // obj.ratio / 2)
    obj.face_rect_center = (obj.image_width // obj.ratio / 2, obj.image_height // obj.ratio / 2)
    obj.ROI_center = (obj.image_width // obj.ratio / 2, obj.image_height // obj.ratio / 2)
    obj.Image_loaded = False
    obj.Pupil_ROI_exist = False
    obj.Face_ROI_exist = False
    obj.eye_corner_mode = False
    obj.eyecorner = None
    obj.eye_corner_center = None
    obj.erased_pixels = None
    obj.mnd = 3
    obj.reflect_brightness = 230
    obj.binary_threshold = 220
    obj.Show_binary = False
    obj.clustering_method = "SimpleContour"
    obj.saturation_method = "None"
    obj.binary_method = "Adaptive"
    obj.primary_direction = None
    obj.secondary_direction = None


class SaturationSettings:
    def __init__(self,
                 primary_direction=None,
                 brightness_curve=1.0,
                 brightness=1.0,
                 secondary_direction=None,
                 brightness_concave_power=1.5,
                 secondary_BrightGain=1.0,
                 saturation_ununiform = 1):
        self.primary_direction = primary_direction
        self.brightness_curve = brightness_curve
        self.brightness = brightness
        self.secondary_direction = secondary_direction
        self.brightness_concave_power = brightness_concave_power
        self.secondary_BrightGain = secondary_BrightGain
        self.saturation_ununiform = saturation_ununiform

def change_Gradual_saturation(image_bgr: np.ndarray, settings: SaturationSettings):
    cv2.imwrite(r"C:\Users\faezeh.rabbani\FACEIT_DATA\Frames\output_image.png", image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    cv2.imwrite(r"C:\Users\faezeh.rabbani\FACEIT_DATA\Frames\hsv.png", hsv)
    h, w = hsv.shape[:2]
    gradient = np.ones((h, w), dtype=np.float32)

    def get_direction_mask(direction, curve, strength):
        if direction == "Right":
            return np.tile(np.linspace(1.0, strength, w) ** curve, (h, 1))
        elif direction == "Left":
            return np.tile(np.linspace(strength, 1.0, w) ** curve, (h, 1))
        elif direction == "Down":
            return np.tile(np.linspace(1.0, strength, h) ** curve, (w, 1)).T
        elif direction == "UP":
            return np.tile(np.linspace(strength, 1.0, h) ** curve, (w, 1)).T
        else:
            raise ValueError(f"Unknown direction: {direction}")

    # Apply primary direction
    if settings.primary_direction is not None:
        gradient *= get_direction_mask(settings.primary_direction,
         settings.brightness_curve, settings.brightness)


    def get_symmetric_concave_mask(h: int, w: int, direction="Horizontal", strength=2.0, power = 1.5):
        size = w if direction in ["Horizontal"] else h
        x = np.linspace(-1, 1, size)
        curve = np.abs(x) ** power
        scaled = 1 + (strength - 1) * curve
        if direction == "Horizontal":
            return np.tile(scaled, (h, 1))
        elif direction == "Vertical":
            return np.tile(scaled, (w, 1)).T

    if settings.secondary_direction is not None:
        gradient *= get_symmetric_concave_mask(h, w, "Vertical",
          settings.secondary_BrightGain, settings.brightness_concave_power)

    # Apply gradient to brightness channel
    hsv[..., 2] *= gradient
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    # change saturation channel
    hsv[..., 1] *= settings.saturation_ununiform
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

    # Convert back to RGB
    bgr_result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)
    return rgb_result

def change_saturation_uniform(image, saturation=0, contrast=1.0):
    """
    Changes the saturation of an image and adjusts brightness to make dark pixels darker and bright pixels brighter.

    Parameters:
        image (numpy.ndarray): The input image (BGR or grayscale).
        saturation (float): Saturation adjustment value in percentage (e.g., 20 = increase by 20%).
        contrast (float): Contrast multiplier (e.g., 1.2 = increase contrast by 20%).

    Returns:
        numpy.ndarray: The processed image in BGR format.
    """
    brightness = 0

    # If image is grayscale, convert to BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to HSV for saturation manipulation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Apply saturation and brightness changes
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * (1 + saturation / 100.0), 0, 255)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * (1 + saturation / 100.0), 0, 255)

    # Convert back to BGR and apply contrast
    hsv_image = hsv_image.astype(np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    return image



def apply_intensity_gradient_gray(gray_image: np.ndarray, settings) -> np.ndarray:
    """
    Applies intensity gradient and optional uniform saturation adjustment to a grayscale image.

    Parameters:
        gray_image (np.ndarray): Input grayscale image (H, W).
        settings (SaturationSettings): Settings including brightness and optional saturation adjustment.

    Returns:
        np.ndarray: Output grayscale image with applied intensity and saturation (via HSV) modifications.
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")

    height, width = gray_image.shape
    gradient = np.ones((height, width), dtype=np.float32)

    def get_direction_mask(direction, curve, strength):
        if direction == "Right":
            return np.tile(np.linspace(1.0, strength, width) ** curve, (height, 1))
        elif direction == "Left":
            return np.tile(np.linspace(strength, 1.0, width) ** curve, (height, 1))
        elif direction == "Down":
            return np.tile(np.linspace(1.0, strength, height) ** curve, (width, 1)).T
        elif direction == "UP":
            return np.tile(np.linspace(strength, 1.0, height) ** curve, (width, 1)).T
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def get_symmetric_concave_mask(h: int, w: int, direction="Horizontal", strength=2.0, power = 1.5):
        size = w if direction == "Horizontal" else h
        x = np.linspace(-1, 1, size)
        curve = np.abs(x) ** power
        scaled = 1 + (strength - 1) * curve
        if direction == "Horizontal":
            return np.tile(scaled, (h, 1))
        elif direction == "Vertical":
            return np.tile(scaled, (w, 1)).T

    # === Apply directional gradient ===
    if settings.primary_direction:
        gradient *= get_direction_mask(settings.primary_direction, settings.brightness_curve, settings.brightness)

    # === Apply symmetric concave gradient if needed ===
    if settings.secondary_direction:
        gradient *= get_symmetric_concave_mask(height, width, "Vertical", settings.secondary_BrightGain, settings.brightness_concave_power)

    # === Convert grayscale to BGR for HSV conversion ===
    if len(gray_image.shape) == 2 or gray_image.shape[2] == 1:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(gray_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # === Apply uniform saturation adjustment ===
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + settings.saturation_ununiform / 100.0), 0, 255)

    # === Apply brightness/intensity gradient ===
    hsv[..., 2] *= gradient
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)


    # === Convert back to BGR ===
    bgr_result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    bgr_result = cv2.convertScaleAbs(bgr_result, alpha=settings.saturation_ununiform)

    return bgr_result



def show_ROI(ROI, image, ROI_type = "pupil"):
    sub_image = ROI.rect()
    top = int(sub_image.top())*2
    bottom = int(sub_image.bottom())*2
    left = int(sub_image.left())*2
    right = int(sub_image.right())*2
    sub_region = image[top:bottom, left:right]
    frame = [top,bottom, left,right]
    if ROI_type == "pupil":
        height, width = sub_region.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (width // 2, height // 2)
        cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

        # === Generate masked_processed for analysis ===
        if sub_region.ndim == 2:
            final_image = cv2.bitwise_and(sub_region, sub_region, mask=mask)
        elif sub_region.ndim == 3 and sub_region.shape[2] == 3:
            final_image = cv2.bitwise_and(sub_region, sub_region, mask=mask)
        elif sub_region.ndim == 3 and sub_region.shape[2] == 4:
            channels = cv2.split(sub_region)
            for i in range(3):
                channels[i] = cv2.bitwise_and(channels[i], channels[i], mask=mask)
            final_image = cv2.merge(channels)
        else:
            raise ValueError("Unsupported processed image format")
    else:
        final_image = sub_region


    return final_image, frame
def show_ROI2(sub_image, image):

    top = int(sub_image.top())*2
    bottom = int(sub_image.bottom())*2
    left = int(sub_image.left())*2
    right = int(sub_image.right())*2
    sub_region = image[top:bottom, left:right]
    frame = [top,bottom, left,right]
    height, width = sub_region.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # === Generate masked_processed for analysis ===
    if sub_region.ndim == 2:
        masked_processed = cv2.bitwise_and(sub_region, sub_region, mask=mask)
    elif sub_region.ndim == 3 and sub_region.shape[2] == 3:
        masked_processed = cv2.bitwise_and(sub_region, sub_region, mask=mask)
    elif sub_region.ndim == 3 and sub_region.shape[2] == 4:
        channels = cv2.split(sub_region)
        for i in range(3):
            channels[i] = cv2.bitwise_and(channels[i], channels[i], mask=mask)
        masked_processed = cv2.merge(channels)
    else:
        raise ValueError("Unsupported processed image format")

    return masked_processed, frame



def second_region(graphicsView_subImage,graphicsView_MainFig):
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

    # Convert BGR (OpenCV format) to RGB for QImage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create QImage from RGB data
    qimage = QtGui.QImage(image_rgb.data, image_width, image_height, image_rgb.strides[0], QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qimage)

    # Scale the pixmap to fit inside the window
    scaled_pixmap = pixmap.scaled(image_width/2, image_height/2, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
    item.setZValue(-1)
    scene.addItem(item)

    graphicsView_MainFig.setScene(scene)
    scene.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
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
    return image

def load_frame_by_index(cap, index):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index < 0 or index >= total_frames:
        raise IndexError("Index out of range")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Error: Could not read frame at index {index}.")
    return frame


def setup_sliders(parent,min,max,set_value, orientation):
    Slider = QtWidgets.QSlider(parent)
    if orientation == "Vertical":
        Slider.setOrientation(QtCore.Qt.Vertical)
    elif orientation == "horizontal":
        Slider.setOrientation(QtCore.Qt.Horizontal)
    Slider.setMinimum(min)
    Slider.setMaximum(max)
    Slider.setValue(set_value)
    Slider.setEnabled(False)
    return Slider
def set_active_style():
    return """
    QWidget {
        background-color: #3d4242;
        color: #000000;
    }
    
    QRadioButton::indicator:checked {
    background-color: #CD853F;
    border: 1px solid white;
    }

    QRadioButton::indicator:unchecked {
        background-color: #4d5454;
        border: 1px solid gray;
    }

    QLabel,
    QRadioButton,
    QCheckBox,
    QGroupBox::title {
        color: white;
    }

    QPushButton {
        background-color: #CD853F;
        color: white;
        border: 3px outset #CD853F;
        padding: 4px;
    }
    QPushButton:hover {
        background-color: #c24b23;
    }

    QLineEdit, QSlider {
        background-color: #3d4242;
        color: #000000;
        border: 1px solid #3d4242;
        padding: 5px;
    }

    QSlider::groove:horizontal {
        border: 1px solid #999999;
        height: 8px;
        background: #b0b0b0;
        margin: 2px 0;
    }

    QSlider::handle:horizontal {
        background: #CD853F;
        border: 1px ridge #CD853F;
        width: 8px;
        height: 20px;
        margin: -7px 0;
        border-radius: 3px;
    }
    """
def set_inactive_style():
    return """
    
    QWidget {
        background-color: #3d4242;
        color: #000000;
    }
    
    QRadioButton::indicator:checked {
    background-color: 3d4242;
    border: 1px solid white;
    }

    QRadioButton::indicator:unchecked {
        background-color: 3d4242;
        border: 1px solid gray;
    }

    QLabel,
    QRadioButton,
    QCheckBox,
    QGroupBox::title {
        color: white;
    }

    QPushButton {
        background-color:  #3d4242;
        color: red;
        border: 3px outset  #3d4242;
        padding: 4px;
    }
    QPushButton:hover {
        background-color:  #3d4242;
    }

    QLineEdit, QSlider {
        background-color:  #4d5454;
        color: #4d5454;
        border: 1px solid  #4d5454;
        padding: 5px;
    }
    QSlider::groove:horizontal {
        border: 1px solid  #4d5454;
        height: 8px;
        background:  #4d5454;
        margin: 2px 0;
    }
    QSlider::handle:horizontal {
        background: #4d5454;
        border: 1px solid  #4d5454;
        width: 10px;
        height: 20px;
        margin: -7px 0;
        border-radius: 4px;
    }
    """


def add_eyecorner(x_pos , y_pos, scene2, graphicsView_subImage):
    if hasattr(graphicsView_subImage, 'eyecorner') and graphicsView_subImage.eyecorner is not None:
        scene2.removeItem(graphicsView_subImage.eyecorner)
    diameter = 5
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






