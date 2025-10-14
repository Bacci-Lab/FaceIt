import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
from FACEIT_codes.analysis import ProcessHandler
from FACEIT_codes.Save import SaveHandler
from FACEIT_codes.Load_data import LoadData
from FACEIT_codes.Graphical_ROIS import ROIHandler
from FACEIT_codes import functions, display_and_plots
from FACEIT_codes.GUI_Intractions import GUI_Intract
from PyQt5.QtCore import QThread
from FACEIT_codes.Workers import PupilWorker


class FaceMotionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the class that has the process function
        self.process_handler = ProcessHandler(self)
        self.save_handler = SaveHandler(self)
        self.load_handler = LoadData(self)
        self.plot_handler = display_and_plots.PlotHandler(self)
        self.Display_handler = display_and_plots.Display(self)
        self.timer = QtCore.QTimer()
        self.is_playing = False
        self.manual_pupil_mode = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logo_path = os.path.join(project_root, "figures", "Logo_FaceIT.jpg")
        if os.path.exists(logo_path):
            MainWindow.setWindowIcon(QtGui.QIcon(logo_path))
        else:
            print(f"Logo not found at {logo_path}")

        # --- source & processing state ---
        self.Eraser_active = False
        self.NPY = False
        self.video = False

        self.folder_path = None
        self.video_path = None
        self.cap = None

        self.find_grooming_threshold = False
        self.contrast = 1
        self.brightness = 1
        self.brightness_curve = 1
        self.brightness_concave_power = 1.5

        self.len_file = 0
        self.erase_size = 20
        self.ratio = 2
        self.mnd = 3
        self.reflect_brightness = 985
        self.c_value = 5
        self.block_size = 17
        self.clustering_method = "SimpleContour"
        self.binary_method = "Adaptive"
        self.binary_threshold = 220
        self.saturation_method = "None"
        self.Show_binary = False

        self.cap = None
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.Main_V_Layout = QtWidgets.QVBoxLayout(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.setup_menubar(MainWindow)
        self.setup_buttons()
        self.setup_graphics_views()
        self.initiate_sliders()
        self.setup_Result()
        self.setup_styles()
        self.setup_connections()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.roi_handler = ROIHandler(self)
        MainWindow.showMaximized()
        self.PupilROIButton.clicked.connect(
            lambda: self.execute_pupil_roi() if self.NPY or self.video else self.warning("Load data to analyse"))
        self.FaceROIButton.clicked.connect(
            lambda: self.execute_face_roi() if self.NPY or self.video else self.warning("Load data to analyse"))

        self.ReflectionButton.clicked.connect(lambda: self.execute_reflect_roi())


    def execute_reflect_roi(self):
        self.roi_handler.Add_ROI(
            roi_type='reflection',
            roi_center=self.reflection_center,
            image=self.image,
            height=self.reflect_height,
            width=self.reflect_width,
            color='gray',
            handle_size=10
        )


    def execute_pupil_roi(self):
        self.roi_handler.Add_ROI(
            roi_type='pupil',
            roi_center=self.ROI_center,
            image=self.image,
            height=50,
            width=80,
            handle_size=10,
            color='palevioletred',
            Button=self.ReflectionButton,
            Button2=self.Erase_Button,
            Button3=self.PupilROIButton,
            Button4=self.Process_Button,
            Button5=self.Add_eyecorner,
            checkBox_pupil=self.checkBox_pupil
        )
        self.Pupil_ROI_exist = True
    def execute_face_roi(self):
        self.roi_handler.Add_ROI(
            roi_type='face',
            roi_center=self.ROI_center,
            image = self.image,
            height=50,
            width=80,
            handle_size=10,
            color='coral',
            Button=None,
            Button2=None,
            Button3=self.FaceROIButton,
            Button4=self.Process_Button,
            Button5=None,
            checkBox_face=self.checkBox_face)
        self.Face_ROI_exist = True


    def setup_menubar(self, MainWindow):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.File_menue = self.menubar.addMenu("File")
        self.setting_menue = self.menubar.addMenu("setting")
        self.LoadVideo = QtWidgets.QAction("Load video", MainWindow)
        self.LoadVideo.setShortcut("Ctrl+v")
        self.load_np = QtWidgets.QAction("Load numpy images", MainWindow)
        self.load_np.setShortcut("Ctrl+n")
        self.LoadProcessedData = QtWidgets.QAction("Load Processed Data", MainWindow)
        self.File_menue.addAction(self.LoadVideo)
        self.File_menue.addAction(self.load_np)
        self.File_menue.addAction(self.LoadProcessedData)
        self.open_settings_action = QtWidgets.QAction("Open Settings Window", MainWindow)
        self.setting_menue.addAction(self.open_settings_action)
        self.open_settings_action.triggered.connect(self.open_settings_window)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

    def open_settings_window(self):
        self.settings_window = QtWidgets.QWidget()
        self.settings_window.setWindowTitle("Settings")
        self.settings_window.resize(400, 300)
        main_layout = QtWidgets.QVBoxLayout()

        # Create a layout for the Brush Size slider
        brush_size_layout = QtWidgets.QHBoxLayout()
        brush_size_label = QtWidgets.QLabel("Brush Size:")
        self.brush_size_edit = QtWidgets.QLineEdit(str(self.erase_size))
        self.brush_size_edit.setReadOnly(True)
        self.brush_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(40)
        self.brush_size_slider.setValue(self.erase_size)
        self.brush_size_slider.valueChanged.connect(self.get_brush_size_value)
        brush_size_layout.addWidget(brush_size_label)
        brush_size_layout.addWidget(self.brush_size_slider)
        brush_size_layout.addWidget(self.brush_size_edit)
        main_layout.addLayout(brush_size_layout)
        cluster_pixel_layout = QtWidgets.QHBoxLayout()
        cluster_pixel_label = QtWidgets.QLabel("MND:")
        self.mnd_edit = QtWidgets.QLineEdit(str(self.mnd))
        self.mnd_edit.setReadOnly(True)
        self.mnd_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mnd_slider.setMinimum(1)
        self.mnd_slider.setMaximum(30)
        self.mnd_slider.setValue(self.mnd)
        self.mnd_slider.valueChanged.connect(self.get_cluster_pixel_value)
        #--------------------------------------- Reflect Bar ------------------------#
        br_reflect_layout = QtWidgets.QHBoxLayout()
        br_reflect_label = QtWidgets.QLabel("Reflect br:")
        self.reflect_brightness_edit = QtWidgets.QLineEdit(str(self.reflect_brightness))
        self.reflect_brightness_edit.setReadOnly(True)
        self.reflect_brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.reflect_brightness_slider.setMinimum(10)
        self.reflect_brightness_slider.setMaximum(999)
        self.reflect_brightness_slider.setSingleStep(1)
        self.reflect_brightness_slider.setValue(self.reflect_brightness)
        self.reflect_brightness_slider.valueChanged.connect(self.get_reflect_brightness_value)
        #------------------------------------  C ---------------------------------------#
        C_layout = QtWidgets.QHBoxLayout()
        C_label = QtWidgets.QLabel("C")
        self.C_edit = QtWidgets.QLineEdit(str(self.c_value))
        self.C_edit.setReadOnly(True)
        self.C_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.C_slider.setMinimum(1)
        self.C_slider.setMaximum(20)
        self.C_slider.setValue(self.c_value)
        self.C_slider.valueChanged.connect(self.get_c_value)
        #------------------------------------ bllock_size ---------------------------------------#
        block_size_layout = QtWidgets.QHBoxLayout()
        block_size_label = QtWidgets.QLabel("block size")
        self.block_size_edit = QtWidgets.QLineEdit(str(self.block_size))
        self.block_size_edit.setReadOnly(True)
        self.block_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.block_size_slider.setMinimum(1)
        self.block_size_slider.setMaximum(40)
        self.block_size_slider.setValue(self.block_size)
        self.block_size_slider.valueChanged.connect(self.get_block_value)
        #-----------------------------------------------------------------------------------#
        cluster_pixel_layout.addWidget(cluster_pixel_label)
        cluster_pixel_layout.addWidget(self.mnd_slider)
        cluster_pixel_layout.addWidget(self.mnd_edit)
        br_reflect_layout.addWidget(br_reflect_label)
        br_reflect_layout.addWidget(self.reflect_brightness_slider)
        br_reflect_layout.addWidget(self.reflect_brightness_edit)
        #-------------------------------------
        C_layout.addWidget(C_label)
        C_layout.addWidget(self.C_slider)
        C_layout.addWidget(self.C_edit)
        block_size_layout.addWidget(block_size_label)
        block_size_layout.addWidget(self.block_size_slider)
        block_size_layout.addWidget(self.block_size_edit)
        main_layout.addLayout(cluster_pixel_layout)
        main_layout.addLayout(br_reflect_layout)
        main_layout.addLayout(C_layout)
        main_layout.addLayout(block_size_layout)
        self.settings_window.setLayout(main_layout)
        self.settings_window.show()

    def get_brush_size_value(self):
        self.erase_size = self.brush_size_slider.value()
        self.brush_size_edit.setText(str(self.erase_size))
        return self.erase_size

    def get_reflect_brightness_value(self):
        slider_value = self.reflect_brightness_slider.value()
        self.reflect_brightness = slider_value
        self.reflect_brightness_edit.setText(f"{self.reflect_brightness:.1f}")  # format with 1 decimal
        return self.reflect_brightness

    def get_c_value(self):
        self.c_value = self.C_slider.value()
        self.C_edit.setText((str(self.c_value)))
        return self.c_value

    def get_block_value(self):
        self.block_size = self.block_size_slider.value()
        self.block_size_edit.setText((str(self.block_size)))
        return self.block_size


    def get_cluster_pixel_value(self):
        self.mnd = self.mnd_slider.value()
        self.mnd_edit.setText((str(self.mnd)))
        return self.mnd
    def setup_Result(self):
        self.vertical_process_Layout = QtWidgets.QVBoxLayout()
        self.graphicsView_whisker = QtWidgets.QGraphicsView(self.centralwidget)
        self.vertical_process_Layout.addWidget(self.graphicsView_whisker)
        self.graphicsView_pupil = QtWidgets.QGraphicsView(self.centralwidget)
        self.vertical_process_Layout.addWidget(self.graphicsView_pupil)
        self.slider_layout = QtWidgets.QHBoxLayout()
        self.Slider_frame = functions.setup_sliders(self.centralwidget, 0, self.len_file, 0, "horizontal")
        self.Slider_frame.setEnabled(False)
        self.slider_layout.addWidget(self.Slider_frame)
        self.lineEdit_frame_number = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_frame_number.setFixedWidth(50)
        self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))
        self.slider_layout.addWidget(self.lineEdit_frame_number)
        self.vertical_process_Layout.addLayout(self.slider_layout)
        self.Main_V_Layout.addLayout(self.vertical_process_Layout)

    def change_cursor_color(self):
        if hasattr(self, 'motion_energy'):
            self.find_grooming_threshold = True
        else:
            pass

    def setup_graphics_views(self):
        self.Image_H_Layout = QtWidgets.QHBoxLayout()
        self.Image_H_Layout.addWidget(self.groupBox)
        self.graphicsView_MainFig = GUI_Intract(self.centralwidget)
        self.graphicsView_MainFig.parent = self
        self.Image_H_Layout.addWidget(self.graphicsView_MainFig)
        self.graphicsView_subImage = GUI_Intract(self.centralwidget)
        self.graphicsView_subImage.parent = self
        self.graphicsView_subImage.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_subImage.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.Image_H_Layout.addWidget(self.graphicsView_subImage)
        self.Main_V_Layout.addLayout(self.Image_H_Layout)

    def setup_buttons(self):
        # Create a group box to hold all the widgets
        self.groupBox = QtWidgets.QGroupBox()
        self.groupBoxLayout = QtWidgets.QVBoxLayout(self.groupBox)  # Layout for the group box

        # Main Layout
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)

        # === ROI Tools ===
        self.roiLayout = QtWidgets.QGridLayout()
        self.groupBoxLayout.addWidget(QtWidgets.QLabel("ROI Tools"))
        self.groupBoxLayout.addLayout(self.roiLayout)

        self.PupilROIButton = QtWidgets.QPushButton("Pupil ROI")
        self.roiLayout.addWidget(self.PupilROIButton, 0, 0)

        self.FaceROIButton = QtWidgets.QPushButton("Face ROI")
        self.roiLayout.addWidget(self.FaceROIButton, 0, 1)

        self.Add_eyecorner = QtWidgets.QPushButton("Add Eye Corner")
        self.Add_eyecorner.setEnabled(False)
        self.roiLayout.addWidget(self.Add_eyecorner, 0, 2)

        self.ReflectionButton = QtWidgets.QPushButton("Add Reflection")
        self.ReflectionButton.setEnabled(False)
        self.roiLayout.addWidget(self.ReflectionButton, 1, 0)

        self.Erase_Button = QtWidgets.QPushButton("Eraser")
        self.Erase_Button.setEnabled(False)
        self.roiLayout.addWidget(self.Erase_Button, 1, 1)

        self.Undo_Erase_Button = QtWidgets.QPushButton("Undo Erase")
        self.roiLayout.addWidget(self.Undo_Erase_Button, 1, 2)

        # === Detection Tools ===
        self.detectionLayout = QtWidgets.QGridLayout()
        self.groupBoxLayout.addWidget(QtWidgets.QLabel("Post processing"))
        self.groupBoxLayout.addLayout(self.detectionLayout)

        self.detect_blinking_Button = QtWidgets.QPushButton("Detect blinking")
        self.detectionLayout.addWidget(self.detect_blinking_Button, 0, 0)

        self.Filtering_pupil_Button = QtWidgets.QPushButton("Filtering pupil")
        self.detectionLayout.addWidget(self.Filtering_pupil_Button, 0, 1)

        self.grooming_Button = QtWidgets.QPushButton("Define Grooming threshold")
        self.detectionLayout.addWidget(self.grooming_Button, 1, 0)

        self.Undo_grooming_Button = QtWidgets.QPushButton("Undo Grooming")
        self.detectionLayout.addWidget(self.Undo_grooming_Button, 1, 1)

        self.Undo_blinking_Button = QtWidgets.QPushButton("Undo blinking\Filtering detection")
        self.detectionLayout.addWidget(self.Undo_blinking_Button, 0, 2)

        # === File Operations ===
        self.fileOpsLayout = QtWidgets.QVBoxLayout()
        self.groupBoxLayout.addWidget(QtWidgets.QLabel("File Operations"))
        self.groupBoxLayout.addLayout(self.fileOpsLayout)

        self.Process_Button = QtWidgets.QPushButton("Process")
        self.Process_Button.setEnabled(False)
        self.fileOpsLayout.addWidget(self.Process_Button)

        self.Save_Button = QtWidgets.QPushButton("Save")
        self.Save_Button.setEnabled(False)
        self.fileOpsLayout.addWidget(self.Save_Button)


        self.PlayPause_Button = QtWidgets.QPushButton("Play/Stop")
        self.PlayPause_Button.setEnabled(False)
        self.fileOpsLayout.addWidget(self.PlayPause_Button)


        # === Options and Threshold ===
        self.optionsLayout = QtWidgets.QGridLayout()

        label = QtWidgets.QLabel("Options & Threshold")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        self.groupBoxLayout.addWidget(label)
        self.groupBoxLayout.addLayout(self.optionsLayout)


        # Clustering method
        self.Clustering_Label = QtWidgets.QLabel("Clustering Method:")
        self.optionsLayout.addWidget(self.Clustering_Label, 0, 0)

        self.radioButton_DBSCAN = QtWidgets.QRadioButton("DBSCAN")
        self.optionsLayout.addWidget(self.radioButton_DBSCAN, 1, 0)

        self.radioButton_watershed = QtWidgets.QRadioButton("Watershed")
        self.optionsLayout.addWidget(self.radioButton_watershed, 2, 0)

        self.radioButton_SimpleContour = QtWidgets.QRadioButton("Simple Contour")
        self.optionsLayout.addWidget(self.radioButton_SimpleContour, 3, 0)
        self.radioButton_SimpleContour.setChecked(True)

        #------------------------------------------------

        # === SECOND COLUMN: Binary & Face ===
        self.checkBox_binary = QtWidgets.QCheckBox("Show Binary")
        self.optionsLayout.addWidget(self.checkBox_binary, 0, 2)

        self.checkBox_face = QtWidgets.QCheckBox("Whisker Pad")
        self.checkBox_face.setEnabled(False)
        self.optionsLayout.addWidget(self.checkBox_face, 1, 2)


        # === THIRD COLUMN: Pupil & Save Video ===
        self.checkBox_pupil = QtWidgets.QCheckBox("Pupil")
        self.checkBox_pupil.setEnabled(False)
        self.optionsLayout.addWidget(self.checkBox_pupil, 2, 2)

        self.checkBox_nwb = QtWidgets.QCheckBox("Save NWB")
        self.optionsLayout.addWidget(self.checkBox_nwb, 0, 3)

        self.save_video = QtWidgets.QCheckBox("Save Video")
        self.optionsLayout.addWidget(self.save_video, 1, 3)

        # Grooming Threshold
        self.grooming_limit_Label = QtWidgets.QLabel("Grooming Threshold:")
        self.optionsLayout.addWidget(self.grooming_limit_Label, 2, 3)

        self.lineEdit_grooming_y = QtWidgets.QLineEdit()
        self.lineEdit_grooming_y.setFixedWidth(50)
        self.optionsLayout.addWidget(self.lineEdit_grooming_y,3, 3)

        # Final add
        self.mainLayout.addWidget(self.groupBox)
        self.centralwidget.setLayout(self.mainLayout)

    ###################################
    def create_slider_block(self, label_text, slider, line_edit):
        block_layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignLeft)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(slider)
        line_edit.setFixedWidth(50)
        slider_layout.addWidget(line_edit)

        block_layout.addWidget(label)
        block_layout.addLayout(slider_layout)
        return block_layout

    def initiate_sliders(self):
        """Sets up sliders and controls in a 3-column layout with frames."""
        self.slider_grid = QtWidgets.QGridLayout()

        # === LEFT COLUMN: Uniform Settings ===
        self.uniform_groupbox = QtWidgets.QGroupBox("Uniform Image Adjustments")
        self.uniform_layout = QtWidgets.QVBoxLayout(self.uniform_groupbox)

        self.radio_button_Uniform = QtWidgets.QRadioButton("Uniform Image Adjustments")
        self.radio_button_Gradual = QtWidgets.QRadioButton("Gradual Image Adjustments")
        self.adjustment_mode_group = QtWidgets.QButtonGroup()
        self.adjustment_mode_group.addButton(self.radio_button_Uniform)
        self.adjustment_mode_group.addButton(self.radio_button_Gradual)

        self.saturation_Slider = functions.setup_sliders(self.centralwidget, 0, 100, 0, "horizontal")
        self.lineEdit_satur_value = QtWidgets.QLineEdit(self.centralwidget)
        saturation_block = self.create_slider_block("Saturation", self.saturation_Slider, self.lineEdit_satur_value)

        self.contrast_Slider = functions.setup_sliders(self.centralwidget, 0, 30, 10, "horizontal")
        self.lineEdit_contrast_value = QtWidgets.QLineEdit(self.centralwidget)
        contrast_block = self.create_slider_block("Contrast", self.contrast_Slider, self.lineEdit_contrast_value)

        self.uniform_layout.addWidget(self.radio_button_Uniform)
        self.uniform_layout.addLayout(saturation_block)
        self.uniform_layout.addLayout(contrast_block)

        self.slider_grid.addWidget(self.uniform_groupbox, 0, 0)

        # === MIDDLE + RIGHT COLUMN: Gradual Settings Inside GroupBox ===
        self.gradual_groupbox = QtWidgets.QGroupBox("Gradual Image Adjustments")
        self.gradual_groupbox_layout = QtWidgets.QVBoxLayout(self.gradual_groupbox)

        # MIDDLE COLUMN
        self.middle_column_layout = QtWidgets.QVBoxLayout()
        self.primary_light_label = QtWidgets.QLabel("Primary Light Direction")
        self.primary_directions_layout = QtWidgets.QHBoxLayout()

        self.radio_button_none = QtWidgets.QRadioButton("None")
        self.radio_button_up = QtWidgets.QRadioButton("Up")
        self.radio_button_down = QtWidgets.QRadioButton("Down")
        self.radio_button_left = QtWidgets.QRadioButton("Left")
        self.radio_button_right = QtWidgets.QRadioButton("Right")
        self.radio_button_none.setChecked(True)

        for btn in [self.radio_button_none, self.radio_button_up, self.radio_button_down,
                    self.radio_button_left, self.radio_button_right]:
            self.primary_directions_layout.addWidget(btn)

        self.radio_group_primary_direction = QtWidgets.QButtonGroup()
        for btn in [self.radio_button_none, self.radio_button_up, self.radio_button_down,
                    self.radio_button_left, self.radio_button_right]:
            self.radio_group_primary_direction.addButton(btn)
            btn.toggled.connect(self.update_light_direction)

        self.brightness_curve_Slider = functions.setup_sliders(self.centralwidget, 0, 30, 15, "horizontal")
        self.lineEdit_brightness_curve_value = QtWidgets.QLineEdit(self.centralwidget)
        brightness_curve_block = self.create_slider_block("Primary Brightness Curve",
                                                          self.brightness_curve_Slider,
                                                          self.lineEdit_brightness_curve_value)

        self.BrightGain_primary_Slider = functions.setup_sliders(self.centralwidget, 10, 30, 10, "horizontal")
        self.lineEdit_brightGain_primary_value = QtWidgets.QLineEdit(self.centralwidget)
        primary_brightGain_block = self.create_slider_block("Primary Brightness Gain",
                                                    self.BrightGain_primary_Slider,
                                                    self.lineEdit_brightGain_primary_value)

        self.middle_column_layout.addWidget(self.radio_button_Gradual)
        self.middle_column_layout.addWidget(self.primary_light_label)
        self.middle_column_layout.addLayout(self.primary_directions_layout)
        self.middle_column_layout.addLayout(brightness_curve_block)
        self.middle_column_layout.addLayout(primary_brightGain_block)

        # RIGHT COLUMN
        self.right_column_layout = QtWidgets.QVBoxLayout()
        self.secondary_light_label = QtWidgets.QLabel("Secondary Light Direction")
        self.secondary_directions_layout = QtWidgets.QHBoxLayout()

        self.radio_button_none_secondary = QtWidgets.QRadioButton("None")
        self.radio_button_H = QtWidgets.QRadioButton("Horizontal")
        self.radio_button_V = QtWidgets.QRadioButton("Vertical")
        self.radio_button_none_secondary.setChecked(True)

        for btn in [self.radio_button_none_secondary, self.radio_button_H,
                 self.radio_button_V]:
            self.secondary_directions_layout.addWidget(btn)

        self.radio_group_secondary_direction = QtWidgets.QButtonGroup()
        for btn in [self.radio_button_none_secondary, self.radio_button_H,
                    self.radio_button_V]:
            self.radio_group_secondary_direction.addButton(btn)
            btn.toggled.connect(self.update_secondary_light_direction)

        self.brightness_concave_power_Slider = functions.setup_sliders(self.centralwidget, -10, 30, 10, "horizontal")
        self.lineEdit_brightness_concave_power = QtWidgets.QLineEdit(self.centralwidget)
        brightness_concave_power_block = self.create_slider_block("Secondary Brightness Concave Power",
                                                                    self.brightness_concave_power_Slider,
                                                                    self.lineEdit_brightness_concave_power)

        self.brightGain_secondary_Slider = functions.setup_sliders(self.centralwidget, 10, 30, 10, "horizontal")
        self.lineEdit_brightGain_secondary_value = QtWidgets.QLineEdit(self.centralwidget)
        brightGain_secondary_block = self.create_slider_block("Secondary Brightness Gain",
                                                              self.brightGain_secondary_Slider,
                                                              self.lineEdit_brightGain_secondary_value)

        self.right_column_layout.addWidget(self.secondary_light_label)
        self.right_column_layout.addLayout(self.secondary_directions_layout)
        self.right_column_layout.addLayout(brightness_concave_power_block)
        self.right_column_layout.addLayout(brightGain_secondary_block)

        # Combine MIDDLE + RIGHT into one HORIZONTAL layout
        self.middle_right_layout = QtWidgets.QHBoxLayout()
        self.middle_right_layout.addLayout(self.middle_column_layout)
        self.middle_right_layout.addLayout(self.right_column_layout)

        # Add the middle-right layout to gradual groupbox
        self.gradual_groupbox_layout.addLayout(self.middle_right_layout)

        # === Stretch saturation_ununiform_block across whole bottom ===
        self.saturation_ununiform_Slider = functions.setup_sliders(self.centralwidget, 0, 30, 10, "horizontal")
        self.lineEdit_satur_ununiform_value = QtWidgets.QLineEdit(self.centralwidget)
        saturation_ununiform_block = self.create_slider_block("Saturation", self.saturation_ununiform_Slider,
                                                              self.lineEdit_satur_ununiform_value)

        self.saturation_ununiform_container = QtWidgets.QHBoxLayout()
        self.saturation_ununiform_container.addLayout(saturation_ununiform_block)
        self.gradual_groupbox_layout.addLayout(self.saturation_ununiform_container)

        # === Finally add gradual groupbox to the main slider grid ===
        self.slider_grid.addWidget(self.gradual_groupbox, 0, 1)
        self.slider_grid.setColumnStretch(0, 1)
        self.slider_grid.setColumnStretch(1, 2)
        self.Main_V_Layout.addLayout(self.slider_grid)

        # === ADDITIONAL Global Slider BELOW both boxes ===
        self.binary_threshold_slider = functions.setup_sliders(self.centralwidget, 0, 255, 220, "horizontal")
        self.lineEdit_binary_threshold_value = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_binary_threshold_value.setFixedWidth(50)
        self.binary_threshold_slider.setStyleSheet(functions.set_inactive_style())


        self.binary_threshold_block = QtWidgets.QHBoxLayout()
        self.radioButton_Constant = QtWidgets.QRadioButton("Constant Binary")
        self.binary_threshold_block.addWidget(self.radioButton_Constant)
        self.binary_threshold_block.addWidget(self.binary_threshold_slider)
        self.binary_threshold_block.addWidget(self.lineEdit_binary_threshold_value)

        self.gradual_groupbox.setStyleSheet(functions.set_inactive_style())
        self.uniform_groupbox.setStyleSheet(functions.set_inactive_style())

        self.Main_V_Layout.addLayout(self.binary_threshold_block)

    def setup_connections(self):
        self.LoadVideo.triggered.connect(self.load_handler.load_video)
        self.load_np.triggered.connect(self.load_handler.open_image_folder)
        self.saturation_Slider.valueChanged.connect(self.satur_value)
        self.saturation_ununiform_Slider.valueChanged.connect(self.satur_ununiform_value)
        self.brightness_curve_Slider.valueChanged.connect(self.update_brightness_curve)
        self.brightness_concave_power_Slider.valueChanged.connect(self.update_brightness_concave_power)
        self.BrightGain_primary_Slider.valueChanged.connect(self.update_brightness)
        self.brightGain_secondary_Slider.valueChanged.connect(self.update_BrightGain_secondary)
        self.contrast_Slider.valueChanged.connect(self.contrast_value)
        self.binary_threshold_slider.valueChanged.connect(self.update_binary_threshold)
        self.Slider_frame.valueChanged.connect(self.Display_handler.update_frame_view)
        self.lineEdit_frame_number.editingFinished.connect(self.update_slider)
        self.Process_Button.clicked.connect(self.process_handler.process)
        self.Add_eyecorner.clicked.connect(self.eyecorner_clicked)
        self.Undo_blinking_Button.clicked.connect(self.init_undo_blinking)
        self.checkBox_binary.stateChanged.connect(self.update_binary_flag)
        self.detect_blinking_Button.clicked.connect(self.start_blinking_detection)
        self.Filtering_pupil_Button.clicked.connect(self.Filtering_pupil)
        self.Save_Button.clicked.connect(self.save_handler.init_save_data)
        self.grooming_Button.clicked.connect(self.change_cursor_color)
        self.Undo_grooming_Button.clicked.connect(self.undo_grooming)
        self.Erase_Button.clicked.connect(self.graphicsView_subImage.activateEraseMode)
        self.Undo_Erase_Button.clicked.connect(self.graphicsView_subImage.undoBrushStrokes)
        self.radioButton_DBSCAN.toggled.connect(self.update_clustering_method)
        self.radioButton_watershed.toggled.connect(self.update_clustering_method)
        self.radioButton_SimpleContour.toggled.connect(self.update_clustering_method)
        self.radioButton_Constant.toggled.connect(self.update_binary_method)
        self.radio_button_Uniform.toggled.connect(self.update_saturation_method)
        self.radio_button_Gradual.toggled.connect(self.update_saturation_method)
        ########################
        self.PlayPause_Button.clicked.connect(self.toggle_play_pause)
        self.timer.timeout.connect(self.play_next_frame)

    def setup_styles(self):
        self.centralwidget.setStyleSheet(functions.set_active_style())
        self.lineEdit_brightGain_primary_value.setStyleSheet("background-color: #999999")
        self.lineEdit_binary_threshold_value.setStyleSheet("background-color: #999999")
        self.lineEdit_brightness_curve_value.setStyleSheet("background-color: #999999")
        self.lineEdit_brightness_concave_power.setStyleSheet("background-color: #999999")
        self.lineEdit_brightGain_secondary_value.setStyleSheet("background-color: #999999")
        self.lineEdit_frame_number.setStyleSheet("background-color: #999999")
        self.lineEdit_satur_value.setStyleSheet("background-color: #999999")
        self.lineEdit_contrast_value.setStyleSheet("background-color: #999999")
        self.lineEdit_grooming_y.setStyleSheet("background-color: #999999")
        self.lineEdit_satur_ununiform_value.setStyleSheet("background-color: #999999")


    def toggle_play_pause(self):
        if not self.is_playing:
            self.timer.start(2)
            self.PlayPause_Button.setText("Pause")
        else:
            self.timer.stop()
            self.PlayPause_Button.setText("Play")
        self.is_playing = not self.is_playing

    def play_next_frame(self):
        current = self.Slider_frame.value()
        if current < self.Slider_frame.maximum():
            self.Slider_frame.setValue(current + 1)
        else:
            self.timer.stop()
            self.PlayPause_Button.setText("Play")
            self.is_playing = False

    def clear_graphics_view(self, graphicsView):
        """Clear any existing layout or widgets in the graphicsView."""
        if graphicsView.layout() is not None:
            old_layout = graphicsView.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old_layout)


    def set_frame(self, face_frame=None, Pupil_frame=None, reflect_ellipse = None):
        if face_frame is not None:
            self.Face_frame = face_frame
        if Pupil_frame is not None:
            self.Pupil_frame = Pupil_frame
        if reflect_ellipse is not None:
            self.reflect_ellipse = reflect_ellipse


    def pupil_check(self):
        return self.checkBox_pupil.isChecked()
    def update_binary_flag(self, state):
        self.Show_binary = (state == QtCore.Qt.Checked)
    def face_check(self):
        return self.checkBox_face.isChecked()
    def nwb_check(self):
        return self.checkBox_nwb.isChecked()
    def save_video_chack(self):
        return self.save_video.isChecked()

    def update_saturation_method(self):
        if self.radio_button_Uniform.isChecked():
            self.gradual_groupbox.setStyleSheet(functions.set_inactive_style())
            self.uniform_groupbox.setStyleSheet(functions.set_active_style())
            self.saturation_method = "Uniform"
            self.brightness_curve_Slider.setEnabled(False)
            self.BrightGain_primary_Slider.setEnabled(False)
            self.brightGain_secondary_Slider.setEnabled(False)
            self.brightness_concave_power_Slider.setEnabled(False)
            self.saturation_ununiform_Slider.setEnabled(False)
            self.contrast_Slider.setEnabled(True)
            self.saturation_Slider.setEnabled(True)
        elif self.radio_button_Gradual.isChecked():
            self.gradual_groupbox.setStyleSheet(functions.set_active_style())
            self.uniform_groupbox.setStyleSheet(functions.set_inactive_style())
            self.saturation_method = "Gradual"
            self.contrast_Slider.setEnabled(False)
            self.saturation_Slider.setEnabled(False)
            self.brightness_curve_Slider.setEnabled(True)
            self.BrightGain_primary_Slider.setEnabled(True)
            self.brightGain_secondary_Slider.setEnabled(True)
            self.brightness_concave_power_Slider.setEnabled(True)
            self.saturation_ununiform_Slider.setEnabled(True)

    def update_light_direction(self):
        selected_direction = None
        if self.radio_button_none.isChecked():
            selected_direction = None
        if self.radio_button_right.isChecked():
            selected_direction = "Right"
        elif self.radio_button_down.isChecked():
            selected_direction = "Down"
        elif self.radio_button_up.isChecked():
            selected_direction = "UP"
        elif self.radio_button_left.isChecked():
            selected_direction = "Left"

        self.primary_direction = selected_direction

    def update_secondary_light_direction(self):
        selected_direction = None
        if self.radio_button_none_secondary.isChecked():
            selected_direction = None
        elif self.radio_button_H.isChecked():
            selected_direction = "Horizontal"
        elif self.radio_button_V.isChecked():
            selected_direction = "Vertical"
        self.secondary_direction = selected_direction


    def update_clustering_method(self):
        if self.radioButton_DBSCAN.isChecked():
            self.clustering_method = "DBSCAN"
        elif self.radioButton_watershed.isChecked():
            self.clustering_method = "watershed"
        elif self.radioButton_SimpleContour.isChecked():
            self.clustering_method = "SimpleContour"

    def update_binary_method(self):
        if self.radioButton_Constant.isChecked():
            self.binary_method = "Constant"
            self.binary_threshold_slider.setEnabled(True)
            self.binary_threshold_slider.setStyleSheet(functions.set_active_style())
        else:
            self.binary_method = "Adaptive"
            self.binary_threshold_slider.setEnabled(False)
            self.binary_threshold_slider.setStyleSheet(functions.set_inactive_style())



    def satur_value(self, value):
        """Update saturation value and apply changes to the displayed sub-region."""
        self.lineEdit_satur_value.setText(str(value))
        self.saturation = value

        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False

        # Ensure `sub_region` and `scene2` exist before trying to update the display
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()  # Initialize if missing

            # Update the display with the new saturation
            self.Display_handler.display_sub_region(self.sub_region,"pupil", Detect_pupil=True)
    def satur_ununiform_value(self, value):
        self.lineEdit_satur_ununiform_value.setText(str( value/10))
        self.saturation_ununiform = value/10

        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False

        # Ensure `sub_region` and `scene2` exist before trying to update the display
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()  # Initialize if missing

            # Update the display with the new saturation
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)

    def contrast_value(self, value):
        self.lineEdit_contrast_value.setText(str(value/10))
        self.contrast =value/10
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()  # Initialize if missing

            # Update the display with the new saturation
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)

    def update_brightness_curve(self, value):
        self.lineEdit_brightness_curve_value.setText(str(value/10))
        self.brightness_curve = value/10
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)

    def update_brightness_concave_power(self, value):
        self.lineEdit_brightness_concave_power.setText(str(value/10))
        self.brightness_concave_power = value/10
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)


    def update_binary_threshold(self, value):
        self.lineEdit_binary_threshold_value.setText(str(value))
        self.binary_threshold = value
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)




    def update_brightness(self, value):
        self.lineEdit_brightGain_primary_value.setText(str(value/10))
        self.brightness = value/10
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)


    def update_BrightGain_secondary(self, value):
        self.lineEdit_brightGain_secondary_value.setText(str(value/10))
        self.secondary_BrightGain = value/10
        if self.checkBox_binary.isChecked():
            self.Show_binary = True
        else:
            self.Show_binary = False
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()
            self.Display_handler.display_sub_region(self.sub_region, "pupil", Detect_pupil=True)

    def update_slider(self):
        try:
            value = int(self.lineEdit_frame_number.text())
            if 0 <= value <= self.Slider_frame.maximum():
                self.Slider_frame.setValue(value)
            else:
                self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))
        except ValueError:
            self.lineEdit_frame_number.setText(str(self.Slider_frame.value()))

    def start_pupil_dilation_computation(self, images):
        self.sub_image = self.pupil_ROI.rect()
        self.thread = QThread()
        self.worker = PupilWorker(
            images, self.process_handler, self.saturation, self.contrast,
            self.erased_pixels,self.brightness_concave_power,self.secondary_direction, self.reflect_ellipse, self.mnd, self.reflect_brightness,
            self.clustering_method, self.binary_method, self.binary_threshold, self.saturation_method, self.saturation_ununiform, self.primary_direction,
            self.brightness, self.brightness_curve, self.secondary_BrightGain, self.c_value, self.block_size, self.sub_image)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_pupil_results)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.handle_worker_error)
        self.thread.start()



    def handle_pupil_results(self, result):
        (self.pupil_dilation,
         self.pupil_center_X,
         self.pupil_center_y,
         self.pupil_center,
         self.X_saccade,
         self.Y_saccade,
         self.pupil_distance_from_corner,
         self.width,
         self.height,
         self.frame_pos,
         self.frame_center,
         self.frame_axes,
         self.angle
         ) = result

        self.final_pupil_area = self.pupil_dilation
        self.X_saccade_updated = self.X_saccade
        self.Y_saccade_updated = self.Y_saccade

        self.plot_handler.plot_result(
            self.pupil_dilation,
            self.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.X_saccade
        )

    def handle_worker_error(self, error_msg):
        self.warning(f"Pupil processing error: {error_msg}")

    def start_blinking_detection(self):
        if hasattr(self, 'pupil_dilation'):
            self.blinking_ids = self.process_handler.detect_blinking(self.width, self.height, self.pupil_dilation, self.X_saccade,
                                                                     self.Y_saccade)
        else:
            self.warning("Process Pupil first")


    def display_removed_grooming(self, grooming_thr, facemotion ):
        self.facemotion_without_grooming, self.grooming_ids, self.grooming_thr = self.process_handler.remove_grooming(grooming_thr, facemotion)
        self.plot_handler.plot_result(self.facemotion_without_grooming, self.graphicsView_whisker, "motion")

    def undo_grooming(self):
        if hasattr(self, 'motion_energy'):
            self.plot_handler.plot_result(self.motion_energy, self.graphicsView_whisker, "motion")
            self.facemotion_without_grooming = self.motion_energy

    def init_undo_blinking(self):
        if hasattr(self, 'pupil_dilation'):
            self.blinking = np.full((len(self.pupil_dilation),), np.nan)
            self.Undo_blinking()
        else:
            self.warning("Process Pupil first")

    def Undo_blinking(self):
        if hasattr(self, 'pupil_dilation') and self.pupil_dilation is not None:
            self.final_pupil_area = np.array(self.pupil_dilation)
        else:
            self.warning("Process Pupil first")

        self.X_saccade_updated = np.array(self.X_saccade)
        self.Y_saccade_updated = np.array(self.Y_saccade)
        self.plot_handler.plot_result(self.pupil_dilation, self.graphicsView_pupil, "pupil", color="palegreen",
                         saccade=self.X_saccade)
    def Filtering_pupil(self):
        if hasattr(self, 'pupil_dilation'):
            self.blinking_ids = self.process_handler.Pupil_Filtering(self.pupil_dilation, self.X_saccade, self.Y_saccade)
        else:
            self.warning("Process Pupil first")
    def eyecorner_clicked(self):
        self.eye_corner_mode = True
        self.Eraser_active = False


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

def main():
    """Entry point for the FACEIT command-line execution."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
