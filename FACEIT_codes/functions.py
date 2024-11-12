from tqdm import tqdm
import os.path
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from datetime import datetime

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap

def initialize_attributes(obj, image):
    obj.image_height, obj.image_width =  image.shape
    obj.reflection_center = (obj.image_width // 2, obj.image_height // 2)
    obj.reflect_height = 15
    obj.reflect_width = 15
    obj.Face_frame = None
    obj.Pupil_frame = None
    obj.sub_region = None
    obj.ROI_center = (obj.image_width // 2, obj.image_height // 2)
    obj.blank_R_center = (obj.image_width // 2, obj.image_height // 2)
    obj.blank_height = 15
    obj.blank_width = 15
    obj.blank_ellipse = None
    obj.reflect_ellipse = None
    obj.saturation = 0
    obj.frame = None
    obj.pupil_ROI = None
    obj.face_ROI = None
    obj.pupil_detection = None
    obj.pupil_ellipse_items = None
    obj.current_ROI = None
    obj.ROI_exist = False
    obj.oval_center = (obj.image_width // 2, obj.image_height // 2)
    obj.face_rect_center = (obj.image_width // 2, obj.image_height // 2)
    obj.ROI_center = (obj.image_width // 2, obj.image_height // 2)
    obj.Image_loaded = False
    obj.Pupil_ROI_exist = False
    obj.Face_ROI_exist = False
    obj.eye_corner_mode = False
    obj.eyecorner = None
    obj.eye_corner_center = None



def enable_Button(Button):
    Button.setEnabled(True)
def disable_Button(Button):
    Button.setEnabled(False)
def Add_ROI(scene,scene2,image,graphicsView_MainFig,graphicsView_subImage, ROI_center, ROI_type, reflect_height,
            reflect_width,blank_height,blank_width, blank_center = None,
             Button= None,Button2 = None,Button3 = None,Button4 = None, Button5=None, reflection_center=None,save_path = None):

    saturation = 0
    if ROI_type == 'pupil':
        pupil_ROI, pupil_handles = draw_ROI(ROI_center, 'pupil', 50, 80,10, color='palevioletred')
        scene.addItem(pupil_ROI)
        for handle in pupil_handles.values():
            scene.addItem(handle)
        graphicsView_MainFig.pupil_handles = pupil_handles
        graphicsView_MainFig.pupil_ROI = pupil_ROI

        sub_region, Pupil_frame = show_ROI(pupil_ROI, image)
        _ =display_sub_region(graphicsView_subImage, sub_region, scene2, ROI_type,saturation, save_path)
        enable_Button(Button)
        enable_Button(Button2)
        enable_Button(Button4)
        enable_Button(Button5)
        disable_Button(Button3)


    elif ROI_type == 'face':
        face_ROI, face_handles = draw_ROI(ROI_center, 'face', 50, 80,10, color='palegreen')
        scene.addItem(face_ROI)
        for handle in face_handles.values():
            scene.addItem(handle)
        graphicsView_MainFig.face_handles = face_handles
        graphicsView_MainFig.face_ROI = face_ROI
        ####################################################
        sub_region, Face_frame = show_ROI(face_ROI, image)
        _ = display_sub_region(graphicsView_subImage, sub_region, scene2,ROI_type,saturation, save_path = None)
        disable_Button(Button3)
        enable_Button(Button4)

    elif ROI_type == 'reflection':
        reflect_ROI, reflect_handles = draw_ROI(reflection_center, 'reflection', reflect_height, reflect_width, 3 , color='gray')
        scene2.addItem(reflect_ROI)
        for handle in reflect_handles.values():
            scene2.addItem(handle)
        graphicsView_subImage.reflect_ROIs.append(reflect_ROI)
        graphicsView_subImage.reflect_handles_list.append(reflect_handles)
        graphicsView_subImage.Reflect_centers.append(reflection_center)
        graphicsView_subImage.reflect_heights.append(reflect_height)
        graphicsView_subImage.reflect_widths.append(reflect_width)
        ########################################################
    elif ROI_type == 'blank':

        blank_ROI, blank_handles = draw_ROI(blank_center, 'blank', blank_height,
                                                                       blank_width,3, color='black')
        scene2.addItem(blank_ROI)
        for handle in blank_handles.values():
            scene2.addItem(handle)
        graphicsView_subImage.blank_ROIs.append(blank_ROI)
        graphicsView_subImage.blank_handles_list.append(blank_handles)
        graphicsView_subImage.blank_centers.append(blank_center)
        graphicsView_subImage.blank_heights.append(blank_height)
        graphicsView_subImage.blank_widths.append(blank_width)
        #####################################################
    elif ROI_type == 'pupil_detection':
        pupil_detection, _ = draw_ROI(ROI_center, 'pupil_detection', 50, 80,10, color='red')
        scene.addItem(pupil_detection)
        graphicsView_MainFig.pupil_detection = pupil_detection
        ####################################################
        sub_region, Face_frame = show_ROI(pupil_detection, image)
        _ = display_sub_region(graphicsView_subImage, sub_region, scene2,ROI_type,saturation, save_path = None)

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

def draw_ROI(oval_center, ROI_type, ROI_height,ROI_width ,handle_size, color= 'gold'):
    if ROI_type == 'pupil':
        color2 ='teal'
        ROI = QtWidgets.QGraphicsEllipseItem(oval_center[0] - ROI_width / 2,
                                                 oval_center[1] - ROI_height / 2,
                                                 ROI_width,
                                                 ROI_height)
    elif ROI_type == 'face':
        color2 = 'teal'
        ROI = QtWidgets.QGraphicsRectItem(oval_center[0] - ROI_width / 2,
                                                 oval_center[1] - ROI_height / 2,
                                                 ROI_width,
                                                 ROI_height)
    elif ROI_type == 'reflection':
        color2 = 'gray'
        ROI = QtWidgets.QGraphicsEllipseItem(oval_center[0] - ROI_width / 2,
                                                 oval_center[1] - ROI_height / 2,
                                                 ROI_width,
                                                 ROI_height)
    elif ROI_type == 'blank':
        color2 = 'blue'
        ROI = QtWidgets.QGraphicsRectItem(oval_center[0] - ROI_width / 2,
                                                 oval_center[1] - ROI_height / 2,
                                                 ROI_width,
                                                 ROI_height)
    elif ROI_type == "pupil_detection":
        color2 = 'red'
        ROI = QtWidgets.QGraphicsEllipseItem(oval_center[0] - ROI_width / 2,
                                                 oval_center[1] - ROI_height / 2,
                                                 ROI_width,
                                                 ROI_height)


    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidth(0)
    ROI.setPen(pen)

    handle_size = handle_size
    handles = {
        'right': QtWidgets.QGraphicsRectItem(oval_center[0] + ROI_width // 2 - handle_size // 2, oval_center[1] - handle_size // 2,
                           handle_size, handle_size)

    }
    handle_pen = QtGui.QPen(QtGui.QColor(color2))
    handle_pen.setWidth(0)
    for handle in handles.values():
        handle.setPen(handle_pen)

    return ROI, handles

def show_ROI(ROI, image):
    sub_image = ROI.rect()
    top = int(sub_image.top())
    bottom = int(sub_image.bottom())
    left = int(sub_image.left())
    right = int(sub_image.right())
    sub_region = image[top:bottom, left:right]
    frame = [top,bottom, left,right]
    return sub_region, frame

# def motion_Energy_comput(direction, frame):
#     Motion_energy = []
#     file_list = sorted([f for f in os.listdir(direction) if f.endswith('.npy')])
#     previous_ROI = None
#     for i, file_name in enumerate(tqdm(file_list, desc="Processing files")):
#         current_array = np.load(os.path.join(direction, file_name), allow_pickle=True)
#         current_ROI = current_array[frame[0]:frame[1], frame[2]:frame[3]]
#         current_ROI = current_ROI.flatten()
#         if previous_ROI is not None:
#             motionEnergyI = np.mean((current_ROI - previous_ROI)**2)
#             Motion_energy.append(motionEnergyI)
#         previous_ROI = current_ROI
#     return Motion_energy


def change_saturation(image, saturation_scale):
    if saturation_scale == 0:
        return image
    else:
        saturation_scale = float(saturation_scale)
        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Scale the saturation channel
        hsv_image[..., 2] = np.clip(hsv_image[..., 2].astype(np.float32) + saturation_scale, 0, 255).astype(np.uint8)
        # Convert back to BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image
import pupil_detection

def detect_pupil(chosen_frame_region, blank_ellipse, reflect_ellipse):
    sub_region_2Dgray = cv2.cvtColor(chosen_frame_region, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(sub_region_2Dgray, 200, 255, cv2.THRESH_BINARY_INV)
    # Blank added
    if blank_ellipse is not None:
        for variable in range(len(blank_ellipse[1])):
            X = blank_ellipse[0][variable][0]
            Y = blank_ellipse[0][variable][1]
            W =  blank_ellipse[1][variable]
            H = blank_ellipse[2][variable]
            Top_left = (int(X - W//2), int(Y + H//2))
            Bottom_right = (int(X + W//2) , int(Y - H//2))
            cv2.rectangle(binary_image,Top_left, Bottom_right, 0, -1)

    binary_image = pupil_detection.find_claster(binary_image)

    if reflect_ellipse is not None:
        All_reflects = [
            [reflect_ellipse[0][variable], (reflect_ellipse[1][variable], reflect_ellipse[2][variable]), 0]
            for variable in
            range(len(reflect_ellipse[1]))]
    else:
        All_reflects = None

    for i in range(4):
        pupil_ROI0, center, width, height, angle = pupil_detection.find_ellipse(binary_image)
        binary_image_update = pupil_detection.overlap_reflect(All_reflects, pupil_ROI0, binary_image)
        binary_image = binary_image_update

    pupil_area = np.pi * (width*height)
    return pupil_ROI0, center, width, height, angle, pupil_area


def display_sub_region(graphicsView, sub_region, scene2, ROI, saturation, save_path, blank_ellipse = None,
                       reflect_ellipse = None, pupil_ellipse_items = None, Detect_pupil = False):
    if pupil_ellipse_items is not None:
        scene2.removeItem(pupil_ellipse_items)
    for item in scene2.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            scene2.removeItem(item)
            del item

    height, width = sub_region.shape[:2]

    if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
        sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)
    sub_region = change_saturation(sub_region, saturation)
    # Add alpha channel to sub_region
    sub_region_rgba = cv2.cvtColor(sub_region, cv2.COLOR_BGR2BGRA)

    if save_path:
        cv2.imwrite(save_path, sub_region_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        np.save(save_path, sub_region_rgba)

    bytes_per_line = width * 4
    qimage = QtGui.QImage(sub_region_rgba.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    pixmap = QPixmap.fromImage(qimage)
    item = QtWidgets.QGraphicsPixmapItem(pixmap)
    if ROI == "pupil":
        item.setZValue(-1)
    scene2.addItem(item)
    if Detect_pupil == True:
        pupil_ROI0, P_detected_center, P_detected_width, P_detected_height, angle, _ = detect_pupil(sub_region_rgba, blank_ellipse, reflect_ellipse)
        pupil_ellipse_item = QtWidgets.QGraphicsEllipseItem(int(P_detected_center[0] - P_detected_width), int(P_detected_center[1] - P_detected_height),
                                                            P_detected_width*2, P_detected_height*2)

        pupil_ellipse_item.setTransformOriginPoint(int(P_detected_center[0]),
                                                   int(P_detected_center[1]))  # Set the origin point for rotation
        pupil_ellipse_item.setRotation(np.degrees(angle))
        print("this is pupil angel test", np.degrees(angle))
        pen = QtGui.QPen(QtGui.QColor(89, 141, 81))
        pen.setWidth(1)
        pen.setStyle(QtCore.Qt.DashLine)
        pupil_ellipse_item.setPen(pen)
        scene2.addItem(pupil_ellipse_item)
        pupil_ellipse_items = pupil_ellipse_item

    scene2.setSceneRect(0, 0, width, height)
    if graphicsView:
        graphicsView.setScene(scene2)
        graphicsView.fitInView(scene2.sceneRect(), QtCore.Qt.KeepAspectRatio)
    return pupil_ellipse_items

def second_region(graphicsView_subImage,graphicsView_MainFig,  image_width, image_height):
    scene2 = QtWidgets.QGraphicsScene(graphicsView_subImage)
    graphicsView_subImage.setScene(scene2)
    graphicsView_subImage.setFixedSize(image_width, image_height)
    graphicsView_MainFig.graphicsView_subImage = graphicsView_subImage
    return scene2

def display_region(image,graphicsView_MainFig, image_width, image_height, scene = None):
    if scene is None:
        scene = QtWidgets.QGraphicsScene(graphicsView_MainFig)
    else:
        for item in scene.items():
            if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                scene.removeItem(item)
                del item


    qimage = QtGui.QImage(image.data, image_width, image_height, QtGui.QImage.Format_Grayscale8)
    pixmap = QtGui.QPixmap.fromImage(qimage)

    item = QtWidgets.QGraphicsPixmapItem(pixmap)
    item.setZValue(-1)
    scene.addItem(item)
    graphicsView_MainFig.setScene(scene)
    scene.setSceneRect(0, 0, image_width, image_height)
    graphicsView_MainFig.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    graphicsView_MainFig.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    graphicsView_MainFig.setFixedSize(image_width, image_height)
    return graphicsView_MainFig, scene


def load_npy_by_index(folder_path, index, image_height = 384):
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if index < 0 or index >= len(npy_files):
        raise IndexError("Index out of range")
    file_path = os.path.join(folder_path, npy_files[index])
    image = np.load(file_path)
    original_height, original_width = image.shape
    aspect_ratio = original_width / original_height
    image_width = int(image_height * aspect_ratio)
    image = cv2.resize(image, (image_width, image_height), interpolation = cv2.INTER_AREA)
    return image


import cv2


def load_frame_by_index(video_path, index, image_height=384):
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
        raise ValueError(f"Error: Could not read frame at index {index}.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    image_width = int(image_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    cap.release()
    return resized_frame

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
    QProgressBar {
        border: 2px solid #999999;
        border-radius: 5px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #cc9900;
        width: 20px;
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

def save_nwb():


    # Step 1: Create an NWBFile object with the required metadata
    nwbfile = NWBFile(
        session_description='Pupil dilation experiment',  # Description of the experiment
        identifier='pupil_dilation_data',  # Unique ID for this session
        session_start_time=datetime.now(),  # Time when the session started
        file_create_date=datetime.now(),  # Creation time of the file
    )

    # Step 2: Add pupil dilation data (as a time series)
    # Replace this with your actual pupil dilation data
    pupil_dilation = np.random.rand(1000)  # Dummy data for pupil dilation
    saccade_data = np.random.rand(1000)  # Dummy data for saccade

    # Time in seconds
    time_stamps = np.arange(0, 1000, 1) * 0.001  # Example time stamps in seconds

    # Create TimeSeries objects for pupil dilation and saccade
    from pynwb.base import TimeSeries

    pupil_dilation_series = TimeSeries(
        name='Pupil Dilation',
        data=pupil_dilation,
        unit='arbitrary units',  # Replace with appropriate unit (e.g., 'mm' for pupil diameter)
        timestamps=time_stamps
    )

    saccade_series = TimeSeries(
        name='Saccade',
        data=saccade_data,
        unit='arbitrary units',  # Replace with appropriate unit
        timestamps=time_stamps
    )

    # Add TimeSeries data to the NWBFile
    nwbfile.add_acquisition(pupil_dilation_series)
    nwbfile.add_acquisition(saccade_series)

    # Step 3: Write the NWB file to disk
    output_filename = 'pupil_dilation_data.nwb'
    with NWBHDF5IO(output_filename, 'w') as io:
        io.write(nwbfile)

    print(f"Data successfully saved to {output_filename}")


