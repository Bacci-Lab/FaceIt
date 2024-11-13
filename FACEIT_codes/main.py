from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMessageBox
import functions
import os.path
import numpy as np
import os
from analysis import ProcessHandler
import bottleneck as bn
import math
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
import pupil_detection
save_path = r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\FaceCamera-imgs\check\sub_region.png"

class CustomGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.setMouseTracking(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.sub_region = None
        self.dragging = False
        self.dragging_face = False
        self.eye_corner_mode = False
        self.dragging_pupil = False
        self.Resizing = False
        self.Resize_face = False
        self.Resize_pupil = False
        self.face_ROI = None
        self.eyecorner = None
        self.pupil_ROI = None
        self.eye_corner_center = None
        self.oval_width = 100
        self.rect_width = 100
        self.ROI_width = 100
        self.oval_height = 50
        self.ROI_height = 50
        self.rect_height = 50
        self.offset = QtCore.QPoint()
        self.scene_pos = None
        self.previous_mouse_pos = None
        self.Face_frame = None
        self.Pupil_frame = None
        self.current_ROI = None
        #----------------------------------- initiate reflection-----------------------------------
        self.Resize_reflect = False
        self.dragging_reflect = False
        self.reflect_ROI = None
        self.reflect_ROIs = []
        self.reflect_handles_list = []
        self.reflect_widths = []
        self.reflect_heights = []
        self.Reflect_centers = []
        self.reflect_ellipse = None
        #------------------------------------ initiate blank---------------------------------------
        self.dragging_blank = False
        self.Resize_blank = False
        self.blank_ROI = None
        self.blank_ROIs = []
        self.blank_handles_list = []
        self.blank_heights = []
        self.blank_widths = []
        self.blank_centers = []
        self.blank_ellipse = None
        self.All_blanks = None
        self.pupil_ellipse_items = None

    def showContextMenu(self, pos):
        context_menu = QtWidgets.QMenu(self)
        delete_action = context_menu.addAction("Delete ROI")
        action = context_menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.scene_pos = self.mapToScene(pos)
            self.delete(self.scene_pos)



    def delete(self, scene_pos):
        for idx, blank_ROI in enumerate(self.blank_ROIs):
            blank_handle = self.blank_handles_list[idx]
            if blank_ROI.contains(scene_pos):
                self.scene().removeItem(blank_ROI)
                self.scene().removeItem(blank_handle['right'])
                del self.blank_ROIs[idx]
                del self.blank_handles_list[idx]
                del self.blank_heights[idx]
                del self.blank_widths[idx]
                del self.blank_centers[idx]
                break

        for idx, reflect_ROI in enumerate(self.reflect_ROIs):
            reflect_handle = self.reflect_handles_list[idx]
            if reflect_ROI.contains(scene_pos):
                self.scene().removeItem(reflect_ROI)
                self.scene().removeItem(reflect_handle['right'])
                del self.reflect_ROIs[idx]
                del self.reflect_handles_list[idx]
                del self.Reflect_centers[idx]
                del self.reflect_widths[idx]
                del self.reflect_heights[idx]
                break

    def mousePressEvent(self, event):

        self.scene_pos = self.mapToScene(event.pos())
        if event.button() == QtCore.Qt.RightButton:
            return
        if hasattr(self.parent, 'eye_corner_mode'):
            if self.parent.eye_corner_mode:
                self.parent.eye_corner_center = functions.add_eyecorner(self.scene_pos.x(),self.scene_pos.y(),
                                                                             self.parent.scene2, self.parent.graphicsView_subImage)
                self.parent.eye_corner_mode = False




        if self.pupil_ROI:
            for handle_name, handle in self.pupil_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_pupil = True

                    self.previous_mouse_pos_pupil = (event.pos().x(), event.pos().y())
                    return

            if self.pupil_ROI.contains(self.scene_pos):
                self.parent.current_ROI = "pupi"
                self.dragging = True
                self.dragging_pupil = True
                self.previous_mouse_pos_pupil = (event.pos().x(), event.pos().y())

                return

        if self.face_ROI:
            for handle_name, handle in self.face_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_face = True

                    self.previous_mouse_pos_face = (event.pos().x(), event.pos().y())
                    return

            if self.face_ROI.contains(self.scene_pos):
                self.parent.current_ROI = "face"
                self.dragging = True
                self.dragging_face = True
                self.previous_mouse_pos_face = (event.pos().x(), event.pos().y())

                return

        for idx, self.reflect_ROI in enumerate(self.reflect_ROIs):
            reflect_handles = self.reflect_handles_list[idx]
            for handle_name, handle in reflect_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_reflect = True

                    self.current_reflect_idx = idx
                    self.previous_mouse_pos_reflect = (self.scene_pos .x(), self.scene_pos .y())
                    return

            if self.reflect_ROI.contains(self.scene_pos):
                self.dragging = True
                self.dragging_reflect = True

                self.current_reflect_idx = idx
                self.previous_mouse_pos_reflect = (self.scene_pos .x(), self.scene_pos .y())

        #---------------------------------------blank ------------------------------------------
        for idx, self.blank_ROI in enumerate(self.blank_ROIs):
            blank_handles = self.blank_handles_list[idx]
            for handle_name, handle in blank_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_blank = True

                    self.current_blank_idx = idx
                    self.previous_mouse_pos_blank = (self.scene_pos .x(), self.scene_pos .y())
                    return

            if self.blank_ROI.contains(self.scene_pos):
                self.dragging = True
                self.dragging_blank = True

                self.current_blank_idx = idx
                self.previous_mouse_pos_blank = (self.scene_pos .x(), self.scene_pos .y())



        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            if self.dragging_face:
                handle_type = "face"
                previous_mouse_pos = self.previous_mouse_pos_face
                self.ROI_center = self.parent.face_rect_center
                self.ROI_width = self.rect_width
                self.ROI_height = self.rect_height
                frame_height_boundary =  self.parent.image_height
                frame_width_boundary = self.parent.image_width

            elif self.dragging_pupil:
                handle_type = "pupil"
                previous_mouse_pos = self.previous_mouse_pos_pupil
                self.ROI_center = self.parent.oval_center
                self.ROI_width = self.oval_width
                self.ROI_height = self.oval_height
                frame_height_boundary = self.parent.image_height
                frame_width_boundary = self.parent.image_width

            elif self.dragging_reflect:
                handle_type = 'reflection'
                previous_mouse_pos = self.previous_mouse_pos_reflect
                self.ROI_center = self.Reflect_centers[self.current_reflect_idx]
                self.ROI_height = self.reflect_heights[self.current_reflect_idx]
                self.ROI_width = self.reflect_widths[self.current_reflect_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]


            elif self.dragging_blank:
                handle_type = 'blank'
                previous_mouse_pos = self.previous_mouse_pos_blank
                self.ROI_center = self.blank_centers[self.current_blank_idx]
                self.ROI_height = self.blank_heights[self.current_blank_idx]
                self.ROI_width = self.blank_widths[self.current_blank_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]

            x = previous_mouse_pos[0]
            y = previous_mouse_pos[1]

            new_pos = self.mapToScene(event.pos())
            self.x_offset = new_pos.x() - x
            self.y_offset = new_pos.y() - y
            # Boundary checks
            half_width = self.ROI_width / 2
            half_height = self.ROI_height / 2
            center_x = self.ROI_center[0] + self.x_offset
            center_y = self.ROI_center[1] + self.y_offset

            if center_x >= frame_width_boundary - half_width:
                center_x = frame_width_boundary - half_width
            elif center_x <= half_width:
                center_x = half_width
            if center_y >= frame_height_boundary - half_height:
                center_y = frame_height_boundary - half_height
            elif center_y <= half_height:
                center_y = half_height

            self.updateEllipse(center_x, center_y, self.ROI_width, self.ROI_height, handle_type)
            if self.dragging_face:
                self.previous_mouse_pos_face = (new_pos.x(), new_pos.y())
                self.parent.face_rect_center = (center_x, center_y)
            elif self.dragging_pupil:
                self.previous_mouse_pos_pupil = (new_pos.x(), new_pos.y())
                self.parent.oval_center = (center_x, center_y)
            elif self.dragging_reflect:
                self.previous_mouse_pos_reflect = (new_pos.x(), new_pos.y())
                self.Reflect_centers[self.current_reflect_idx] = (center_x, center_y)
            elif self.dragging_blank:
                self.previous_mouse_pos_blank = (new_pos.x(), new_pos.y())
                self.blank_centers[self.current_blank_idx] = (center_x, center_y)



        elif self.Resizing:
            if self.Resize_face:
                handle_type = "face"
                previous_mouse_pos = self.previous_mouse_pos_face
                self.ROI_center = self.parent.face_rect_center
                self.ROI_width = self.rect_width
                self.ROI_height = self.rect_height
                frame_width_boundary = self.parent.image_width
                minimum_w_h = 10

            elif self.Resize_pupil:
                handle_type = "pupil"
                previous_mouse_pos = self.previous_mouse_pos_pupil
                self.ROI_center = self.parent.oval_center
                self.ROI_width = self.oval_width
                self.ROI_height = self.oval_height
                frame_width_boundary = self.parent.image_width
                minimum_w_h = 10

            elif self.Resize_reflect:
                handle_type = "reflection"
                previous_mouse_pos = self.previous_mouse_pos_reflect
                self.ROI_center = self.Reflect_centers[self.current_reflect_idx]
                self.ROI_width = self.reflect_widths[self.current_reflect_idx]
                self.ROI_height = self.reflect_heights[self.current_reflect_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]
                minimum_w_h = 1

            elif self.Resize_blank:
                handle_type = "blank"
                previous_mouse_pos = self.previous_mouse_pos_blank
                self.ROI_center = self.blank_centers[self.current_blank_idx]
                self.ROI_width = self.blank_widths[self.current_blank_idx]
                self.ROI_height = self.blank_heights[self.current_blank_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]
                minimum_w_h = 1

            x = previous_mouse_pos[0]
            y = previous_mouse_pos[1]
            new_pos = self.mapToScene(event.pos())
            self.x_offset = new_pos.x() - x
            self.y_offset = new_pos.y() - y
            resized_width = self.ROI_width + 2 * (self.x_offset)
            resized_height = self.ROI_height - 2 * (self.y_offset)

            # Ensure minimum size constraints
            if resized_width < minimum_w_h:
                resized_width = minimum_w_h
            if resized_height < minimum_w_h:
                resized_height = minimum_w_h
            if self.ROI_center[0] + resized_width / 2 >= frame_width_boundary:
                resized_width = (frame_width_boundary - self.ROI_center[0]) * 2
            if self.ROI_center[1] <= resized_height / 2:
                resized_height = self.ROI_center[1] * 2

            self.updateEllipse(self.ROI_center[0], self.ROI_center[1], resized_width, resized_height, handle_type)

            if self.Resize_face:
                self.previous_mouse_pos_face = (new_pos.x(), new_pos.y())
                self.rect_width = resized_width
                self.rect_height = resized_height

            elif self.Resize_pupil:
                self.previous_mouse_pos_pupil = (new_pos.x(), new_pos.y())
                self.oval_width = resized_width
                self.oval_height = resized_height

            elif self.Resize_reflect:
                self.previous_mouse_pos_reflect = (new_pos.x(), new_pos.y())
                self.reflect_heights[self.current_reflect_idx] = resized_height
                self.reflect_widths[self.current_reflect_idx] = resized_width
            elif self.Resize_blank:
                self.previous_mouse_pos_blank = (new_pos.x(), new_pos.y())
                self.blank_heights[self.current_blank_idx] = resized_height
                self.blank_widths[self.current_blank_idx] = resized_width

        super().mouseMoveEvent(event)

    def updateHandles(self, center_x, center_y, handle_type, handle_size):
        half_width = self.ROI_width / 2
        if handle_type == "face":
            handles = self.face_handles
        elif handle_type == "pupil":
            handles = self.pupil_handles
        elif handle_type == "reflection":
            handles = self.reflect_handles_list[self.current_reflect_idx]
        elif handle_type == "blank":
            handles = self.blank_handles_list[self.current_blank_idx]
        handles['right'].setRect(center_x + half_width - handle_size // 2, center_y - handle_size // 2,
                                 handle_size, handle_size)


    def mouseReleaseEvent(self, event):
        if self.dragging:
            if self.dragging_face:
                self.sub_region, self.parent.Face_frame = functions.show_ROI(self.face_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.parent.scene2, "face",self.parent.saturation, save_path = None)
                self.parent.set_frame(self.parent.Face_frame)
            elif self.dragging_pupil:
                self.parent.sub_region, self.parent.Pupil_frame = functions.show_ROI(self.pupil_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.parent.sub_region, self.parent.scene2,"pupil",self.parent.saturation,  save_path = save_path)
                self.parent.set_frame(self.parent.Pupil_frame)
                self.parent.reflection_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)
                self.parent.blank_R_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)


            elif self.dragging_reflect:
                self.reflect_ellipse = [self.Reflect_centers, self.reflect_widths , self.reflect_heights ]
                self.parent.reflect_ellipse = self.reflect_ellipse
            elif self.dragging_blank:
                self.blank_ellipse = [self.blank_centers, self.blank_widths, self.blank_heights]
                self.parent.blank_ellipse = self.blank_ellipse

            self.dragging = False
            self.dragging_face = False
            self.dragging_pupil = False
            self.dragging_reflect = False
            self.dragging_blank = False
        elif self.Resizing:
            if self.Resize_face:
                self.sub_region, self.parent.Face_frame = functions.show_ROI(self.face_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.parent.scene2, "face",self.parent.saturation,  save_path)
                self.parent.set_frame(self.parent.Face_frame)
            elif self.Resize_pupil:
                self.parent.sub_region, self.parent.Pupil_frame = functions.show_ROI(self.pupil_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.parent.sub_region, self.parent.scene2, "pupil",self.parent.saturation, save_path)
                self.parent.set_frame(self.parent.Pupil_frame)
                self.parent.reflection_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)
                self.parent.blank_R_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)
            elif self.Resize_reflect:
                self.reflect_ellipse = [self.Reflect_centers, self.reflect_widths , self.reflect_heights]
                self.parent.reflect_ellipse = self.reflect_ellipse
            elif self.Resize_blank:
                self.blank_ellipse = [self.blank_centers, self.blank_widths , self.blank_heights]
                self.parent.blank_ellipse = self.blank_ellipse
            self.Resizing = False
            self.Resize_pupil = False
            self.Resize_face = False
            self.Resize_reflect = False
            self.Resize_blank = False

        super().mouseReleaseEvent(event)

    def updateEllipse(self, center_x, center_y, width, height, handle_type):
        if handle_type == "face":
            self.face_ROI.setRect(center_x - width / 2, center_y - height / 2, width, height)
            self.updateHandles(center_x, center_y, handle_type, 10)
        elif handle_type == "pupil":
            self.pupil_ROI.setRect(center_x - width / 2, center_y - height / 2, width, height)
            self.updateHandles(center_x, center_y, handle_type, 10)
        elif handle_type == "reflection":
            self.reflect_ROIs[self.current_reflect_idx].setRect(center_x - width / 2, center_y - height / 2, width, height)
            self.reflect_ROIs[self.current_reflect_idx].setBrush(QtGui.QBrush(QtGui.QColor('silver')))
            self.updateHandles(center_x, center_y, handle_type, 3)
        elif handle_type == "blank":
            self.blank_ROIs[self.current_blank_idx].setRect(center_x - width / 2, center_y - height / 2, width, height)
            self.blank_ROIs[self.current_blank_idx].setBrush(QtGui.QBrush(QtGui.QColor('white')))
            self.updateHandles(center_x, center_y, handle_type, 3)

class FaceMotionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the class that has the `process` function
        self.process_handler = ProcessHandler(self)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowIcon(QtGui.QIcon(r"C:\Users\faezeh.rabbani\Downloads\logo.jpg"))
        self.NPY = False
        self.video = False
        self.find_grooming_threshold = False
        self.len_file = 1
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.Main_V_Layout = QtWidgets.QVBoxLayout(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.setup_menubar(MainWindow)
        self.setup_buttons()
        self.setup_graphics_views()
        self.setup_saturation()
        self.setup_Result()
        self.setup_styles()
        self.setup_connections()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.showMaximized()
        self.PupilROIButton.clicked.connect(lambda: self.execute_pupil_roi() if self.NPY or self.video else self.warning("Load data to analyse"))
        self.FaceROIButton.clicked.connect(lambda: self.execute_face_roi() if self.NPY or self.video else self.warning("Load data to analyse"))
        self.ReflectionButton.clicked.connect(lambda: self.execute_reflect_roi())
        self.Add_blank_button.clicked.connect(lambda: self.execute_blank_roi())

    def execute_blank_roi(self):
        functions.Add_ROI(
            self.scene,
            self.scene2,
            self.image,
            self.graphicsView_MainFig,
            self.graphicsView_subImage,
            self.ROI_center, 'blank',
            self.reflect_height,
            self.reflect_width,
            self.blank_height,
            self.blank_width,
            blank_center = self.blank_R_center,
            Button=None,
            Button2=None,
            Button3=None,
            Button4=self.Process_Button,
            Button5=None,
            reflection_center=self.reflection_center)

    def execute_pupil_roi(self):
        functions.Add_ROI(
            self.scene,
            self.scene2,
            self.image,
            self.graphicsView_MainFig,
            self.graphicsView_subImage,
            self.ROI_center,
            'pupil',
            self.reflect_height,
            self.reflect_width,
            self.blank_height,
            self.blank_width,
            Button=self.ReflectionButton,
            Button2=self.Add_blank_button,
            Button3=self.PupilROIButton,
            Button4=self.Process_Button,
            Button5 = self.Add_eyecorner
        )
        self.set_pupil_roi_pressed(True)
    def execute_face_roi(self):
        functions.Add_ROI(
            self.scene,
            self.scene2,
            self.image,
            self.graphicsView_MainFig,
            self.graphicsView_subImage,
            self.ROI_center, 'face',
            self.reflect_height,
            self.reflect_width,
            self.blank_height,
            self.blank_width,
            Button=None,
            Button2=None,
            Button3=self.FaceROIButton,
            Button4=self.Process_Button,
            Button5=None)
        self.set_Face_ROI_pressed(True)
    def execute_reflect_roi(self):
        functions.Add_ROI(
            self.scene,
            self.scene2,
            self.image,
            self.graphicsView_MainFig,
            self.graphicsView_subImage,
            self.ROI_center, 'reflection',
            self.reflect_height,
            self.reflect_width,
            self.blank_height,
            self.blank_width,
            blank_center=self.blank_R_center,
            Button=None,
            Button2=None,
            Button3=None,
            Button4=self.Process_Button,
            Button5=None,
            reflection_center=self.reflection_center)

    def setup_menubar(self, MainWindow):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.File_menue = self.menubar.addMenu("File")
        self.LoadVideo = QtWidgets.QAction("Load video", MainWindow)
        self.LoadVideo.setShortcut("Ctrl+v")
        self.load_np = QtWidgets.QAction("Load numpy images", MainWindow)
        self.load_np.setShortcut("Ctrl+n")
        self.LoadProcessedData = QtWidgets.QAction("Load Processed Data", MainWindow)
        self.File_menue.addAction(self.LoadVideo)
        self.File_menue.addAction(self.load_np)
        self.File_menue.addAction(self.LoadProcessedData)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

    def setup_graphics_views(self):
        self.Image_H_Layout = QtWidgets.QHBoxLayout()
        self.Image_H_Layout.addWidget(self.leftGroupBox)
        self.Image_H_Layout.addWidget(self.rightGroupBox)
        self.graphicsView_MainFig = CustomGraphicsView(self.centralwidget)
        self.graphicsView_MainFig.parent = self
        self.Image_H_Layout.addWidget(self.graphicsView_MainFig)
        self.graphicsView_subImage = CustomGraphicsView(self.centralwidget)
        self.graphicsView_subImage.parent = self
        self.graphicsView_subImage.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_subImage.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.Image_H_Layout.addWidget(self.graphicsView_subImage)
        self.Main_V_Layout.addLayout(self.Image_H_Layout)
    def setup_Result(self):
        self.vertical_process_Layout = QtWidgets.QVBoxLayout()
        self.graphicsView_whisker = QtWidgets.QGraphicsView(self.centralwidget)
        self.vertical_process_Layout.addWidget(self.graphicsView_whisker)
        self.graphicsView_pupil = QtWidgets.QGraphicsView(self.centralwidget)
        self.vertical_process_Layout.addWidget(self.graphicsView_pupil)
        self.slider_layout = QtWidgets.QHBoxLayout()
        self.Slider_frame = functions.setup_sliders(self.centralwidget, 0, self.len_file, 0, "horizontal")
        self.slider_layout.addWidget(self.Slider_frame)
        self.lineEdit_frame_number = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_frame_number.setFixedWidth(50)
        self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))
        self.slider_layout.addWidget(self.lineEdit_frame_number)
        self.vertical_process_Layout.addLayout(self.slider_layout)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.vertical_process_Layout.addWidget(self.progressBar)
        self.Main_V_Layout.addLayout(self.vertical_process_Layout)

    def change_cursor_color(self):
        self.find_grooming_threshold = True
    def setup_buttons(self):
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.leftGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.rightGroupBox = QtWidgets.QGroupBox(self.centralwidget)

        self.mainLayout.addWidget(self.leftGroupBox)
        self.mainLayout.addWidget(self.rightGroupBox)

        self.rightGroupBoxLayout = QtWidgets.QVBoxLayout(self.rightGroupBox)
        #########################################
        self.leftGroupBoxLayout = QtWidgets.QVBoxLayout(self.leftGroupBox)
        # Set the main layout to the central widget
        self.centralwidget.setLayout(self.mainLayout)
        #########

        self.PupilROIButton = QtWidgets.QPushButton("Pupil ROI")
        self.leftGroupBoxLayout.addWidget(self.PupilROIButton)
        self.FaceROIButton = QtWidgets.QPushButton("Face ROI")
        self.leftGroupBoxLayout.addWidget(self.FaceROIButton)
        self.Add_blank_button = QtWidgets.QPushButton("Add blank")
        self.leftGroupBoxLayout.addWidget(self.Add_blank_button)
        self.Add_blank_button.setEnabled(False)
        self.ReflectionButton = QtWidgets.QPushButton("Add Reflection")
        self.leftGroupBoxLayout.addWidget(self.ReflectionButton)
        self.ReflectionButton.setEnabled(False)
        self.Add_eyecorner = QtWidgets.QPushButton("Add Eye corner")
        self.leftGroupBoxLayout.addWidget(self.Add_eyecorner)
        self.Add_eyecorner.setEnabled(False)
        self.Process_Button = QtWidgets.QPushButton("Process")
        self.Process_Button.setEnabled(False)
        self.rightGroupBoxLayout.addWidget(self.Process_Button)
        self.Save_Button = QtWidgets.QPushButton("Save")
        self.Save_Button.setEnabled(False)
        self.rightGroupBoxLayout.addWidget(self.Save_Button)
        self.detect_blinking_Button = QtWidgets.QPushButton("Detect blinking")
        self.rightGroupBoxLayout.addWidget(self.detect_blinking_Button)
        ##############
        self.Undo_blinking_Button = QtWidgets.QPushButton("Undo blinking")
        self.rightGroupBoxLayout.addWidget(self.Undo_blinking_Button)

        self.grooming_Button = QtWidgets.QPushButton("Detect Grooming")
        # self.grooming_Button.setEnabled(False)
        self.rightGroupBoxLayout.addWidget(self.grooming_Button)

        self.Undo_grooming_Button = QtWidgets.QPushButton("Undo Grooming")
        self.rightGroupBoxLayout.addWidget(self.Undo_grooming_Button)
        ##################
        self.exclude_blinking_Button = QtWidgets.QPushButton("exclude blinking")
        self.rightGroupBoxLayout.addWidget(self.exclude_blinking_Button)
        #################################
        self.grooming_limit_Label = QtWidgets.QLabel("grooming threshold")
        self.grooming_limit_Label.setFixedSize(100, 20)
        self.grooming_limit_Label.setStyleSheet("color: white;")
        self.rightGroupBoxLayout.addWidget(self.grooming_limit_Label)
        #########################
        self.lineEdit_grooming_y = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_grooming_y.setFixedWidth(50)
        self.rightGroupBoxLayout.addWidget(self.lineEdit_grooming_y)
        self.checkBox_face = QtWidgets.QCheckBox("Whisker Pad")
        self.leftGroupBoxLayout.addWidget(self.checkBox_face)
        self.checkBox_pupil = QtWidgets.QCheckBox("Pupil")
        self.leftGroupBoxLayout.addWidget(self.checkBox_pupil)
        self.checkBox_nwb = QtWidgets.QCheckBox("Save nwb")
        self.leftGroupBoxLayout.addWidget(self.checkBox_nwb)


    def setup_saturation(self):
        self.sliderLayout = QtWidgets.QVBoxLayout()
        self.saturation_Label = QtWidgets.QLabel("Saturation")
        self.saturation_Label.setAlignment(QtCore.Qt.AlignLeft)
        self.saturation_Label.setStyleSheet("color: white;")
        self.sliderLayout.addWidget(self.saturation_Label)
        self.saturation_slider_layout = QtWidgets.QHBoxLayout()
        self.saturation_Slider = functions.setup_sliders(self.centralwidget, 0, 150, 0, "horizontal")
        self.saturation_slider_layout.addWidget(self.saturation_Slider)
        self.lineEdit_satur_value = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_satur_value.setFixedWidth(50)
        self.saturation_slider_layout.addWidget(self.lineEdit_satur_value)
        self.sliderLayout.addLayout(self.saturation_slider_layout)
        self.Main_V_Layout.addLayout(self.sliderLayout)

    def setup_connections(self):
        self.LoadVideo.triggered.connect(self.Load_video)
        self.saturation_Slider.valueChanged.connect(self.satur_value)
        self.load_np.triggered.connect(self.openImageFolder)
        self.Slider_frame.valueChanged.connect(self.get_np_frame)
        self.lineEdit_frame_number.editingFinished.connect(self.update_slider)
        self.Process_Button.clicked.connect(self.process_handler.process)
        self.Add_eyecorner.clicked.connect(self.eyecorner_clicked)
        self.Undo_blinking_Button.clicked.connect(self.init_undo_blinking)
        self.detect_blinking_Button.clicked.connect(self.start_blinking_detection)
        self.Save_Button.clicked.connect(self.init_save_data)
        self.grooming_Button.clicked.connect(self.change_cursor_color)
        self.Undo_grooming_Button.clicked.connect(self.undo_grooming)



    def setup_styles(self):
        self.centralwidget.setStyleSheet(functions.get_stylesheet())
        functions.set_button_style(self.saturation_Slider, "QSlider")
        functions.set_button_style(self.Slider_frame, "QSlider")
        self.lineEdit_frame_number.setStyleSheet("background-color: #999999")
        self.lineEdit_satur_value.setStyleSheet("background-color: #999999")
        self.lineEdit_grooming_y.setStyleSheet("background-color: #999999")

    def clear_graphics_view(self, graphicsView):
        """Clear any existing layout or widgets in the graphicsView."""
        if graphicsView.layout() is not None:
            old_layout = graphicsView.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old_layout)

    def plot_result(self, data, graphicsView, label, color='#D97A53', saccade=None):
        self.Save_Button.setEnabled(True)
        self.clear_graphics_view(graphicsView)
        # Create the figure and axes
        fig, ax = plt.subplots()
        x_values = np.arange(0, len(data))
        ax.plot(x_values, data, color=color, label=label, linestyle='--')

        # Adjust the plot limits
        data_min = np.min(data)
        data_max = np.max(data)
        range_val = (data_max - data_min)

        # Plot the saccade if available
        if saccade is not None:
            y_values = [data_max + range_val / 10, data_max + range_val / 5]
            ax.pcolormesh(x_values, y_values, saccade, cmap='RdYlGn', shading='flat')

        # Set background and axes properties
        fig.patch.set_facecolor('#3d4242')
        ax.set_facecolor('#3d4242')
        ax.set_xlim(left=0, right=len(data))
        ax.set_ylim(bottom=data_min, top=data_max + range_val / 4)

        # Customize axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)

        # Add legend
        legend = ax.legend(loc='upper right', fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color("white")

        # Handle zooming and panning
        self.setup_interaction_events(fig, ax)

        # Setup canvas and add to the graphics view
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()

        layout = QtWidgets.QVBoxLayout(graphicsView)
        layout.addWidget(canvas)
        graphicsView.setLayout(layout)

        ax.grid(False)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(bottom=0.15)
        canvas.draw()

    def setup_interaction_events(self, fig, ax):
        """Setup zoom and pan events for the plot."""
        self.panning = False
        self.press_event = None

        def create_colored_cursor(color, size=(35, 4)):
            """Create a custom cursor with the specified color."""
            pixmap = QtGui.QPixmap(*size)
            pixmap.fill(color)
            return QtGui.QCursor(pixmap)

        def on_press(event):
            if event.inaxes != ax:
                return
            self.panning = True
            self.press_event = event
            event.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            if self.find_grooming_threshold == True:
                event.canvas.setCursor(create_colored_cursor(QtGui.QColor('blue')))

        def on_motion(event):
            if not self.panning or self.press_event is None or event.xdata is None:
                return
            dx = event.xdata - self.press_event.xdata
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            fig.canvas.draw_idle()

        def on_release(event):
            self.panning = False
            self.press_event = None
            event.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        def on_click(event):
            if self.find_grooming_threshold == True:
                if event.inaxes == ax and event.ydata is not None:
                    self.grooming_thr = event.ydata
                    print(f"Clicked at y: {event.ydata:.2f}")
                    self.find_grooming_threshold = False
                    self.lineEdit_grooming_y.setText(str(int(event.ydata)))
                    self.display_removed_grooming(self.grooming_thr, self.motion_energy)

        def zoom(event):
            current_xlim = ax.get_xlim()
            xdata = event.xdata
            if xdata is None:
                return
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            new_xlim = [xdata - (xdata - current_xlim[0]) * zoom_factor,
                        xdata + (current_xlim[1] - xdata) * zoom_factor]
            ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event', zoom)
        fig.canvas.mpl_connect('button_press_event', on_click)

    def set_frame(self, face_frame=None, Pupil_frame=None, reflect_ellipse = None, blank_ellipse = None):
        if face_frame is not None:
            self.Face_frame = face_frame
        if Pupil_frame is not None:
            self.Pupil_frame = Pupil_frame
        if reflect_ellipse is not None:
            self.reflect_ellipse = reflect_ellipse
        if blank_ellipse is not None:
            self.blank_ellipse = blank_ellipse


    def pupil_check(self):
        return self.checkBox_pupil.isChecked()

    def face_check(self):
        return self.checkBox_face.isChecked()
    def nwb_check(self):
        return self.checkBox_nwb.isChecked()

    def set_pupil_roi_pressed(self, value):
        self.Pupil_ROI_exist = value

    def set_Face_ROI_pressed(self, value):
        self.Face_ROI_exist = value



    def satur_value(self, value):
        self.lineEdit_satur_value.setText(str(self.saturation_Slider.value()))
        self.saturation = value
        if self.sub_region is not None:
            _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.scene2,
                                                 "pupil", self.saturation, save_path = save_path)
        else:
            pass

    def update_slider(self):
        try:
            value = int(self.lineEdit_frame_number.text())
            if 0 <= value <= self.Slider_frame.maximum():
                self.Slider_frame.setValue(value)
            else:
                self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))
        except ValueError:
            self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))


    def init_save_data(self):
        len_data = 100
        if self.pupil_check() == False:
            self.pupil_center = np.full((len_data,), np.nan)
            self.pupil_center_X = np.full((len_data,), np.nan)
            self.pupil_center_y = np.full((len_data,), np.nan)
            self.final_pupil_area = np.full((len_data,), np.nan)
            self.X_saccade_updated = np.full((len_data,), np.nan)
            self.Y_saccade_updated = np.full((len_data,), np.nan)
            self.pupil_distance_from_corner = np.full((len_data,), np.nan)
            self.width = np.full((len_data,), np.nan)
            self.height = np.full((len_data,), np.nan)
        if self.face_check() == False:
            self.motion_energy = np.full((len_data,), np.nan)

        self.save_data(self.pupil_center, self.pupil_center_X, self.pupil_center_y, self.final_pupil_area,
                       self.X_saccade_updated, self.Y_saccade_updated,
                       self.pupil_distance_from_corner, self.width, self.height, self.motion_energy)


    def save_data(self, pupil_center,pupil_center_X, pupil_center_y, pupil_dilation,X_saccade, Y_saccade, pupil_distance_from_corner, width, height,motion_energy ):
        data_dict = {
            'pupil_center': pupil_center,
            'pupil_X_position': pupil_center_X,
            'pupil_Y_position': pupil_center_y,
            'pupil_area': pupil_dilation,
            'X_saccade': X_saccade,
            'Y_saccade': Y_saccade,
            'pupil_distance_from_corner': pupil_distance_from_corner,
            'pupil area width':width,
            'pupil area height': height,
            'motion energy': motion_energy,
        }
        save_directory = os.path.join(self.save_path, "faceit.npz")
        np.savez(save_directory, **data_dict)
        # if self.nwb_check() == True:
        #     self.save_nwb

    def start_pupil_dilation_computation(self, images):
        pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner, width, height =\
            self.process_handler.pupil_dilation_comput(images, self.saturation,self.blank_ellipse, self.reflect_ellipse)
        self.final_pupil_area = pupil_dilation
        self.X_saccade_updated = X_saccade
        self.Y_saccade_updated = Y_saccade
        return pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner,width, height

    def openImageFolder(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print("project_root", project_root)
        default_path = os.path.join(project_root, "test_data", "test_images")
        print("default_path", default_path)

        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder", default_path
        )
        if self.folder_path:
            self.save_path = self.folder_path
            npy_files = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
            self.len_file = len(npy_files)
            self.Slider_frame.setMaximum(self.len_file - 1)
            self.NPY = True
            self.video = False
            self.display_Graphics(self.folder_path)
            self.FaceROIButton.setEnabled(True)
            self.PupilROIButton.setEnabled(True)

    def Load_video(self):
        self.folder_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Video", "",
                                                             "Video Files (*.avi)")
        if self.folder_path:
            directory_path = os.path.dirname(self.folder_path)
            self.save_path = directory_path
            cap = cv2.VideoCapture(self.folder_path)
            self.len_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self.Slider_frame.setMaximum(self.len_file - 1)
            self.video = True
            self.NPY = False
            self.display_Graphics(self.folder_path)
            self.FaceROIButton.setEnabled(True)
            self.PupilROIButton.setEnabled(True)



    def load_image(self, filepath, image_height=384):
        """Load and resize a single image from the given file path."""
        try:
            current_image = np.load(filepath, allow_pickle=True)
            original_height, original_width = current_image.shape
            aspect_ratio = original_width / original_height
            image_width = int(image_height * aspect_ratio)
            resized_image = cv2.resize(current_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            return resized_image
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return None


    def load_images_from_directory(self, directory, image_height=384, max_workers=8):
        """Load images in parallel from the directory with a progress bar update."""
        file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])

        if self.progressBar:
            self.progressBar.setMaximum(len(file_list))

        images = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.load_image, file, image_height): file for file in file_list}

            for i, future in enumerate(as_completed(futures)):
                image = future.result()
                if image is not None:
                    images.append(image)
                if self.progressBar:
                    self.progressBar.setValue(i + 1)

        if self.progressBar:
            self.progressBar.setValue(len(file_list))
        return images
    def display_Graphics(self,folder_path):
        self.frame = 0
        if self.NPY == True:
            self.image = functions.load_npy_by_index(folder_path, self.frame)
        elif self.video == True:
            self.image = functions.load_frame_by_index(folder_path, self.frame)
        functions.initialize_attributes(self, self.image)
        self.scene2 = functions.second_region(self.graphicsView_subImage,
                                                                        self.graphicsView_MainFig, self.image_width, self.image_height)
        self.graphicsView_MainFig, self.scene = functions.display_region \
            (self.image, self.graphicsView_MainFig, self.image_width, self.image_height)

    def start_blinking_detection(self):
        if hasattr(self, 'pupil_dilation'):
            self.process_handler.detect_blinking(self.pupil_dilation, self.width, self.height, self.X_saccade, self.Y_saccade)

        else:
            print("self.pupil_dilation does not exist")
            self.warning("Process Pupil first")
    def display_removed_grooming(self, grooming_thr, facemotion ):
        print(" grooming_thr is ", grooming_thr )
        self.facemotion_without_grooming = self.remove_grooming(grooming_thr, facemotion)
        self.plot_result(self.facemotion_without_grooming, self.graphicsView_whisker, "motion")

    def remove_grooming(self,grooming_thr, facemotion):
        grooming_ids = np.where(facemotion>=grooming_thr)
        facemotion = np.array(facemotion)
        self.facemotion_without_grooming = np.copy(facemotion)
        self.facemotion_without_grooming[grooming_ids] = grooming_thr
        return self.facemotion_without_grooming

    def undo_grooming(self):
        self.plot_result(self.motion_energy, self.graphicsView_whisker, "motion")

    def init_undo_blinking(self):
        if hasattr(self, 'pupil_dilation'):
            self.Undo_blinking()

        else:
            self.warning("Process Pupil first")



    def Undo_blinking(self):
        self.final_pupil_area = np.array(self.pupil_dilation)
        self.X_saccade_updated = np.array(self.X_saccade)
        self.Y_saccade_updated = np.array(self.Y_saccade)
        self.plot_result(self.pupil_dilation, self.graphicsView_pupil, "pupil", color="palegreen",
                         saccade=self.X_saccade)

    def eyecorner_clicked(self):
        self.eye_corner_mode = True
        print("is true", self.eye_corner_mode )

    def load_frames_from_video(self, video_path, max_workers=8, buffer_size=32,image_height=384):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_queue = queue.Queue(maxsize=buffer_size)

        def producer():
            """Producer thread to read frames and put them into the queue."""
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)

            frame_queue.put(None)  # Sentinel to indicate end of frames

        def resize_frame(frame, image_height):
            # Unpack the height, width, and color channels (if they exist)
            if len(frame.shape) == 3:
                original_height, original_width, _ = frame.shape  # For color images
            else:
                original_height, original_width = frame.shape  # For grayscale images

            aspect_ratio = original_width / original_height
            image_width = int(image_height * aspect_ratio)

            # Resize the frame to the new dimensions
            return cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)

        def consumer():
            """Consumer thread to process frames from the queue."""
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                with tqdm(total=total_frames, desc="Processing video frames") as pbar:
                    while True:
                        frame = frame_queue.get()
                        if frame is None:
                            break
                        # Submit resized frame to the thread pool
                        future = executor.submit(resize_frame, frame, image_height)
                        futures.append(future)

                        # Once a future completes, collect the result
                        for future in as_completed(futures):
                            frames.append(future.result())
                            pbar.update(1)
                            futures.remove(future)  # Remove completed future

        # Start producer thread
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # Start consumer thread
        consumer()

        # Ensure all frames are processed and video capture is released
        producer_thread.join()
        cap.release()

        return frames
    ##########################################################################################################
    def get_np_frame(self, frame):
        self.frame = frame
        if self.NPY == True:
            self.image = functions.load_npy_by_index(self.folder_path,
                                                           self.frame)
        elif self.video == True:
            self.image = functions.load_frame_by_index(self.folder_path,
                                                            self.frame)
        self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))
        self.graphicsView_MainFig, self.scene = functions.display_region\
            (self.image,self.graphicsView_MainFig, self.image_width, self.image_height, self.scene)



        if self.Pupil_ROI_exist:
            self.pupil_ROI = self.graphicsView_MainFig.pupil_ROI
            self.sub_region, self.Pupil_frame = functions.show_ROI(self.pupil_ROI, self.image)
            self.pupil_ellipse_items = functions.display_sub_region(self.graphicsView_subImage, self.sub_region,
                                                                            self.scene2,
                                                                            "pupil", self.saturation, save_path,
                                                                            self.blank_ellipse, self.reflect_ellipse,
                                                                            self.pupil_ellipse_items, Detect_pupil=True)
        else:
            if self.Face_ROI_exist:
                self.face_ROI = self.graphicsView_MainFig.face_ROI
                self.sub_region, self.face_ROI = functions.show_ROI(self.face_ROI, self.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region,
                                                         self.scene2,
                                                         "face", self.saturation, save_path,
                                                         self.blank_ellipse, self.reflect_ellipse,
                                                         self.pupil_ellipse_items, Detect_pupil=False
                                                         )
            else:
                pass



    def warning(self, text):
        warning_box = QMessageBox()
        warning_box.setIcon(QMessageBox.Warning)
        warning_box.setWindowTitle("Warning")
        warning_box.setText(text)
        warning_box.setStandardButtons(QMessageBox.Ok)
        warning_box.exec_()





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FaceIt"))
        self.lineEdit_satur_value.setText(_translate("MainWindow", "0"))
        self.lineEdit_grooming_y.setText(_translate("MainWindow", "0"))




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = FaceMotionApp()
        self.ui.setupUi(self)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Window = MainWindow()
    Window.show()
    app.exec_()
