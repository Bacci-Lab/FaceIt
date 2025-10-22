import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


# ======================================================================
# Initialization / state helpers
# ======================================================================

def initialize_attributes(obj, image: np.ndarray) -> None:
    """
    Initialize GUI/app state fields based on an input frame.

    Notes
    -----
    - Sets image height/width depending on whether the frame is gray or color.
    - Initializes all flags and defaults so other code can assume existence.
    """
    if image.ndim == 3:
        obj.image_height, obj.image_width, _ = image.shape
    elif image.ndim == 2:
        obj.image_height, obj.image_width = image.shape
    else:
        raise ValueError("Unsupported image array shape")

    obj.ratio = 2
    obj.reflect_height = 30
    obj.reflect_width = 30

    # Data / frames
    obj.Face_frame = None
    obj.Pupil_frame = None
    obj.sub_region = None
    obj.frame = None

    # Masks / ROI / geometry
    obj.reflect_ellipse = None          # list-like or None
    obj.pupil_ROI = None
    obj.face_ROI = None
    obj.pupil_detection = None
    obj.pupil_ellipse_items = None
    obj.current_ROI = None
    obj.ROI_exist = False

    # Centers (half sized because of display scaling by ratio)
    cx = obj.image_width // obj.ratio / 2
    cy = obj.image_height // obj.ratio / 2
    obj.reflection_center = (cx, cy)
    obj.oval_center = (cx, cy)
    obj.face_rect_center = (cx, cy)
    obj.ROI_center = (cx, cy)

    # UX flags
    obj.Image_loaded = False
    obj.Pupil_ROI_exist = False
    obj.Face_ROI_exist = False
    obj.eye_corner_mode = False
    obj.eyecorner = None
    obj.eye_corner_center = None
    obj.erased_pixels = None

    # Processing defaults
    obj.mnd = 3
    obj.reflect_brightness = 985  # percentile*10 (see detect_reflection_automatically)
    obj.binary_threshold = 220
    obj.Show_binary = False
    obj.clustering_method = "SimpleContour"
    obj.saturation_method = "None"
    obj.binary_method = "Adaptive"

    # Light adjustment
    obj.primary_direction = None
    obj.secondary_direction = None
    obj.saturation = 0                  # uniform saturation (%)
    obj.saturation_ununiform = 1.0      # gradual: multiplicative factor for S
    obj.contrast = 1.0
    obj.brightness = 1.0
    obj.secondary_BrightGain = 1.0
    obj.brightness_curve = 1.0


# ======================================================================
# Light adjustment (Uniform & Gradual)
# ======================================================================

@dataclass
class SaturationSettings:
    """
    Settings for 'Gradual Image Adjustment'.

    primary_direction: "Right" | "Left" | "Down" | "UP" | None
    secondary_direction: "Horizontal" | "Vertical" | None
    brightness_curve: curvature (>1 = steeper) for primary ramp
    brightness: end gain (>=1.0) for primary ramp
    brightness_concave_power: power shaping for symmetric concave mask
    secondary_BrightGain: end gain (>=1.0) for secondary mask
    saturation_ununiform: multiplicative factor for the S channel (>=0)
    """
    primary_direction: Optional[str] = None
    brightness_curve: float = 1.0
    brightness: float = 1.0
    secondary_direction: Optional[str] = None
    brightness_concave_power: float = 1.5
    secondary_BrightGain: float = 1.0
    saturation_ununiform: float = 1.0


def _direction_mask(h: int, w: int, direction: str, curve: float, strength: float) -> np.ndarray:
    if direction == "Right":
        return np.tile(np.linspace(1.0, strength, w) ** curve, (h, 1))
    if direction == "Left":
        return np.tile(np.linspace(strength, 1.0, w) ** curve, (h, 1))
    if direction == "Down":
        return np.tile(np.linspace(1.0, strength, h) ** curve, (w, 1)).T
    if direction == "UP":
        return np.tile(np.linspace(strength, 1.0, h) ** curve, (w, 1)).T
    raise ValueError("Unknown direction: {!r}".format(direction))


def _symmetric_concave_mask(h: int, w: int, axis: str, strength: float, power: float) -> np.ndarray:
    """
    axis: "Horizontal" (vary across width) or "Vertical" (vary across height)
    """
    if axis not in {"Horizontal", "Vertical"}:
        raise ValueError("axis must be 'Horizontal' or 'Vertical'")
    size = w if axis == "Horizontal" else h
    x = np.linspace(-1, 1, size)
    curve = np.abs(x) ** power  # concave outward
    scaled = 1 + (strength - 1) * curve
    return np.tile(scaled, (h, 1)) if axis == "Horizontal" else np.tile(scaled, (w, 1)).T


def change_Gradual_saturation(image_bgr: np.ndarray,
                              settings: SaturationSettings,
                              show: bool = False) -> np.ndarray:
    """
    Apply spatial brightness/saturation gradient in HSV space (color input).

    Returns RGB image (for display). Keeps values in [0, 255].
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = hsv.shape[:2]
    gradient = np.ones((h, w), dtype=np.float32)

    # primary ramp
    if settings.primary_direction:
        gradient *= _direction_mask(h, w,
                                    settings.primary_direction,
                                    settings.brightness_curve,
                                    settings.brightness)

    # secondary symmetric concave gain
    if settings.secondary_direction:
        gradient *= _symmetric_concave_mask(
            h, w,
            axis=settings.secondary_direction,
            strength=settings.secondary_BrightGain,
            power=settings.brightness_concave_power,
        )

    # Apply gradient to V; adjust S multiplicatively
    hsv[..., 2] = np.clip(hsv[..., 2] * gradient, 0, 255)
    hsv[..., 1] = np.clip(hsv[..., 1] * settings.saturation_ununiform, 0, 255)

    # Back to RGB for Qt display
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 2))
        plt.imshow(gradient, cmap="gray", aspect="auto")
        plt.title("Brightness weight mask")
        plt.tight_layout()
        plt.show()

    return rgb


def change_saturation_uniform(image: np.ndarray,
                              saturation: float = 0.0,
                              contrast: float = 1.0) -> np.ndarray:
    """
    Uniform Image Adjustment (color or gray).

    - In HSV: scale S and V by (1 + saturation/100).
    - In BGR: apply contrast gain via convertScaleAbs(alpha=contrast).
    """
    brightness_offset = 0  # keep mean unchanged

    # Ensure BGR input
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gain = 1.0 + saturation / 100.0
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)  # Saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * gain, 0, 255)  # Value (brightness)

    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness_offset)
    return img


def apply_intensity_gradient_gray(gray_image: np.ndarray,
                                  settings: SaturationSettings) -> np.ndarray:
    """
    Gradual adjustment for grayscale input; returns BGR (for display/pipeline).

    Steps
    -----
    1) Build combined gradient (primary + optional secondary).
    2) Convert gray→BGR→HSV, scale S by saturation_ununiform, scale V by gradient.
    3) Back to BGR.
    """
    if gray_image.ndim != 2:
        raise ValueError("apply_intensity_gradient_gray expects a 2D grayscale array")

    h, w = gray_image.shape
    gradient = np.ones((h, w), dtype=np.float32)

    if settings.primary_direction:
        gradient *= _direction_mask(h, w,
                                    settings.primary_direction,
                                    settings.brightness_curve,
                                    settings.brightness)

    if settings.secondary_direction:
        gradient *= _symmetric_concave_mask(
            h, w,
            axis=settings.secondary_direction,
            strength=settings.secondary_BrightGain,
            power=settings.brightness_concave_power,
        )

    bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[..., 1] = np.clip(hsv[..., 1] * (settings.saturation_ununiform), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gradient, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


# ======================================================================
# ROI utilities (de-duplicated)
# ======================================================================

def _ellipse_center_axes_mask(h: int, w: int) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray]:
    """Return center, axes, and a filled elliptical mask for an image of size (h, w)."""
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    return center, axes, mask


def _apply_ellipse_mask(img: np.ndarray) -> np.ndarray:
    """Apply a centered filled ellipse mask to img (gray, BGR, or BGRA)."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360,
                color=255, thickness=-1)

    # Grayscale
    if img.ndim == 2:
        return cv2.bitwise_and(img, img, mask=mask)

    # BGR
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.bitwise_and(img, img, mask=mask)

    # BGRA (preserve alpha channel)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        a   = img[:, :, 3]
        bgr_masked = cv2.bitwise_and(bgr, bgr, mask=mask)
        return np.dstack([bgr_masked, a])

    raise ValueError("Unsupported image format for ellipse masking")


def show_ROI(ROI: QtWidgets.QGraphicsRectItem,
             image: np.ndarray,
             ROI_type: str = "pupil") -> Tuple[np.ndarray, List[int]]:
    """
    Extract and (optionally) mask an ROI from the full frame.

    Returns
    -------
    final_image : np.ndarray
        Sub-image; if ROI_type == 'pupil', a centered elliptical mask is applied.
    frame : [top, bottom, left, right]
    """
    rect = ROI.rect()
    top, bottom = int(rect.top()) * 2, int(rect.bottom()) * 2
    left, right = int(rect.left()) * 2, int(rect.right()) * 2
    sub = image[top:bottom, left:right]
    frame = [top, bottom, left, right]

    if ROI_type == "pupil":
        sub = _apply_ellipse_mask(sub)
    return sub, frame


def show_ROI2(sub_rect: QtCore.QRectF,
              image: np.ndarray) -> Tuple[np.ndarray, List[int], Tuple[int, int], Tuple[int, int]]:
    """
    Variant returning center & axes; kept for compatibility with existing callers.
    """
    top, bottom = int(sub_rect.top()) * 2, int(sub_rect.bottom()) * 2
    left, right = int(sub_rect.left()) * 2, int(sub_rect.right()) * 2
    sub = image[top:bottom, left:right]
    frame = [top, bottom, left, right]

    h, w = sub.shape[:2]
    center, axes, _ = _ellipse_center_axes_mask(h, w)
    masked = _apply_ellipse_mask(sub)
    return masked, frame, center, axes


# ======================================================================
# Graphics / display helpers
# ======================================================================

def second_region(graphicsView_subImage: QtWidgets.QGraphicsView,
                  graphicsView_MainFig: QtWidgets.QGraphicsView) -> QtWidgets.QGraphicsScene:
    """Create a fresh scene for the subimage view and link it to the main view."""
    scene2 = QtWidgets.QGraphicsScene(graphicsView_subImage)
    graphicsView_subImage.setScene(scene2)
    graphicsView_MainFig.graphicsView_subImage = graphicsView_subImage
    return scene2


def display_region(image: np.ndarray,
                   graphicsView_MainFig: QtWidgets.QGraphicsView,
                   image_width: int,
                   image_height: int,
                   scene: Optional[QtWidgets.QGraphicsScene] = None
                   ) -> Tuple[QtWidgets.QGraphicsView, QtWidgets.QGraphicsScene]:
    """
    Display an image (BGR) inside a QGraphicsView; fits and disables scrollbars.

    Notes
    -----
    - If a scene is supplied, removes the previous pixmap item first.
    - Scales to half size (width/2, height/2) while keeping aspect ratio.
    """
    if scene is None:
        scene = QtWidgets.QGraphicsScene(graphicsView_MainFig)
    else:
        # Remove existing pixmap items to avoid stacking
        for item in list(scene.items()):
            if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                scene.removeItem(item)
                del item

    # BGR → RGB for Qt
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QtGui.QImage(image_rgb.data, image_width, image_height,
                          image_rgb.strides[0], QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qimage)
    scaled_pixmap = pixmap.scaled(image_width // 2, image_height // 2,
                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

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


# ======================================================================
# File utils
# ======================================================================

_num_pat = re.compile(r"(\d+)")

def _natural_key(s: str):
    """Natural sort key: frame_2.npy < frame_10.npy."""
    return [int(t) if t.isdigit() else t.lower() for t in _num_pat.split(s)]


def list_npy_files(folder_path: Union[str, Path]) -> List[str]:
    """Return naturally sorted .npy file names in a directory."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError("Expected a directory, got: {!r}".format(folder))
    files = [f.name for f in folder.iterdir() if f.suffix.lower() == ".npy"]
    files.sort(key=_natural_key)
    if not files:
        raise ValueError("No .npy files found in: {!r}".format(folder))
    return files


def load_npy_by_index(folder_path: Union[str, Path],
                      index: int,
                      *,
                      allow_pickle: bool = False,
                      mmap_mode: Optional[str] = None) -> np.ndarray:
    """Load a .npy file by natural order index."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError("load_npy_by_index expects a directory, got: {!r}".format(folder))

    files = list_npy_files(folder)
    if index < 0 or index >= len(files):
        raise IndexError("Index {} out of range (0..{})".format(index, len(files) - 1))

    path = folder / files[index]
    try:
        return np.load(str(path), allow_pickle=allow_pickle, mmap_mode=mmap_mode)
    except Exception as e:
        raise IOError("Failed to load {}: {}".format(path, e)) from e


def load_frame_by_index(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    """Seek to frame `index` and return it; raises if out of range or read fails."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index < 0 or index >= total:
        raise IndexError("Index {} out of range (0..{})".format(index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Could not read frame at index {}.".format(index))
    return frame


# ======================================================================
# UI helpers (styles / sliders / eyecorner marker)
# ======================================================================

def setup_sliders(parent: QtWidgets.QWidget,
                  min_val: int,
                  max_val: int,
                  set_value: int,
                  orientation: str) -> QtWidgets.QSlider:
    """
    Create a disabled slider with given bounds/value.

    orientation: "Vertical" or "Horizontal" (case-insensitive)
    """
    slider = QtWidgets.QSlider(parent)
    if orientation.lower().startswith("v"):
        slider.setOrientation(QtCore.Qt.Vertical)
    else:
        slider.setOrientation(QtCore.Qt.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(set_value)
    slider.setEnabled(False)
    return slider


def set_active_style() -> str:
    """Palette for active widgets."""
    return """
    QWidget { background-color: #3d4242; color: #000000; }

    QRadioButton::indicator:checked { background-color: #CD853F; border: 1px solid white; }
    QRadioButton::indicator:unchecked { background-color: #4d5454; border: 1px solid gray; }

    QLabel, QRadioButton, QCheckBox, QGroupBox::title { color: white; }

    QPushButton {
        background-color: #CD853F; color: white; border: 3px outset #CD853F; padding: 4px;
    }
    QPushButton:hover { background-color: #c24b23; }

    QLineEdit, QSlider { background-color: #3d4242; color: #000000; border: 1px solid #3d4242; padding: 5px; }

    QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #b0b0b0; margin: 2px 0; }
    QSlider::handle:horizontal { background: #CD853F; border: 1px ridge #CD853F; width: 8px; height: 20px;
                                 margin: -7px 0; border-radius: 3px; }
    """


def set_inactive_style() -> str:
    """Palette for inactive/disabled widgets."""
    return """
    QWidget { background-color: #3d4242; color: #000000; }

    QRadioButton::indicator:checked { background-color: #3d4242; border: 1px solid white; }
    QRadioButton::indicator:unchecked { background-color: #3d4242; border: 1px solid gray; }

    QLabel, QRadioButton, QCheckBox, QGroupBox::title { color: white; }

    QPushButton { background-color: #3d4242; color: red; border: 3px outset #3d4242; padding: 4px; }
    QPushButton:hover { background-color: #3d4242; }

    QLineEdit, QSlider { background-color: #4d5454; color: #4d5454; border: 1px solid #4d5454; padding: 5px; }

    QSlider::groove:horizontal { border: 1px solid #4d5454; height: 8px; background: #4d5454; margin: 2px 0; }
    QSlider::handle:horizontal { background: #4d5454; border: 1px solid #4d5454; width: 10px; height: 20px;
                                 margin: -7px 0; border-radius: 4px; }
    """


def add_eyecorner(x_pos: float,
                  y_pos: float,
                  scene2: QtWidgets.QGraphicsScene,
                  graphicsView_subImage: QtWidgets.QGraphicsView) -> Tuple[float, float]:
    """
    Draw/replace a small dot showing the eye corner on the subimage scene.

    Returns the (x, y) tuple for convenience.
    """
    # Remove previous marker if present
    if hasattr(graphicsView_subImage, "eyecorner") and graphicsView_subImage.eyecorner is not None:
        scene2.removeItem(graphicsView_subImage.eyecorner)

    diameter = 5.0
    item = QtWidgets.QGraphicsEllipseItem(x_pos - diameter / 2, y_pos - diameter / 2, diameter, diameter)
    pen = QtGui.QPen(QtGui.QColor("peru"))
    pen.setWidth(0)
    item.setPen(pen)
    item.setBrush(QtGui.QBrush(QtGui.QColor("peru")))

    scene2.addItem(item)
    graphicsView_subImage.eyecorner = item
    return (x_pos, y_pos)
