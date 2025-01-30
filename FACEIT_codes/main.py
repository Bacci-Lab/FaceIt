import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox

# Import modules from the project
from FACEIT_codes.analysis import ProcessHandler
from FACEIT_codes.Save import SaveHandler
from FACEIT_codes.Load_data import LoadData
from FACEIT_codes.Graphical_ROIS import ROIHandler
from FACEIT_codes import functions, display_and_plots
from FACEIT_codes.GUI_Intractions import GUI_Intract
class FaceMotionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the class that has the process function
        self.process_handler = ProcessHandler(self)
        self.save_handler = SaveHandler(self)
        self.load_handler = LoadData(self)
        self.plot_handler = display_and_plots.PlotHandler(self)
        self.Display_handler = display_and_plots.Display(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")

        # Determine the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Construct the path to the logo within the repository
        logo_path = os.path.join(project_root, "figures", "Logo_FaceIT.jpg")

        # Check if the file exists
        if os.path.exists(logo_path):
            # Set the window icon using the logo from the repository
            MainWindow.setWindowIcon(QtGui.QIcon(logo_path))
        else:
            print(f"Logo not found at {logo_path}")

        self.Eraser_active = False
        self.NPY = False
        self.video = False
        self.find_grooming_threshold = False
        self.len_file = 1
        self.erase_size = 5
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
            handle_size=3
        )


    def execute_pupil_roi(self):
        self.roi_handler.Add_ROI(
            roi_type='pupil',
            roi_center=self.ROI_center,
            image=self.image,
            height= 50,
            width = 80,
            handle_size=10,
            color='palevioletred',
            Button=self.ReflectionButton,
            Button2=self.Erase_Button,
            Button3=self.PupilROIButton,
            Button4=self.Process_Button,
            Button5 = self.Add_eyecorner
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
            Button5=None)
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
        # Store the window as an instance attribute to prevent garbage collection
        self.settings_window = QtWidgets.QWidget()
        self.settings_window.setWindowTitle("Settings")
        self.settings_window.resize(400, 300)

        # Create the main layout for the settings window
        main_layout = QtWidgets.QVBoxLayout()

        # Create a layout for the Brush Size slider
        brush_size_layout = QtWidgets.QHBoxLayout()
        brush_size_label = QtWidgets.QLabel("Brush Size:")
        self.brush_size_edit = QtWidgets.QLineEdit("1")
        self.brush_size_edit.setReadOnly(True)
        self.brush_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(10)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.get_brush_size_value)
        brush_size_layout.addWidget(brush_size_label)
        brush_size_layout.addWidget(self.brush_size_slider)
        brush_size_layout.addWidget(self.brush_size_edit)
        main_layout.addLayout(brush_size_layout)

        # Create a layout for the Cluster Pixel line edit
        cluster_pixel_layout = QtWidgets.QHBoxLayout()
        cluster_pixel_label = QtWidgets.QLabel("Cluster Pixel:")
        cluster_pixel_edit = QtWidgets.QLineEdit()
        cluster_pixel_layout.addWidget(cluster_pixel_label)
        cluster_pixel_layout.addWidget(cluster_pixel_edit)
        main_layout.addLayout(cluster_pixel_layout)

        # Set the main layout for the settings window
        self.settings_window.setLayout(main_layout)
        self.settings_window.show()

    def get_brush_size_value(self):
        self.erase_size = self.brush_size_slider.value()
        self.brush_size_edit.setText(str(self.erase_size))
        return self.erase_size

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

        self.detect_blinking_Button = QtWidgets.QPushButton("Detect Blinking")
        self.detectionLayout.addWidget(self.detect_blinking_Button, 0, 0)

        self.Undo_blinking_Button = QtWidgets.QPushButton("Undo Blinking")
        self.detectionLayout.addWidget(self.Undo_blinking_Button, 0, 1)

        self.grooming_Button = QtWidgets.QPushButton("Define Grooming threshold")
        self.detectionLayout.addWidget(self.grooming_Button, 1, 0)

        self.Undo_grooming_Button = QtWidgets.QPushButton("Undo Grooming")
        self.detectionLayout.addWidget(self.Undo_grooming_Button, 1, 1)

        # self.exclude_blinking_Button = QtWidgets.QPushButton("Exclude Blinking")
        # self.detectionLayout.addWidget(self.exclude_blinking_Button, 1, 2)

        # === File Operations ===
        self.fileOpsLayout = QtWidgets.QVBoxLayout()
        self.groupBoxLayout.addWidget(QtWidgets.QLabel("File Operations"))
        self.groupBoxLayout.addLayout(self.fileOpsLayout)

        self.Process_Button = QtWidgets.QPushButton("Process")
        self.Process_Button.setEnabled(False)
        self.fileOpsLayout.addWidget(self.Process_Button)

        self.Save_Button = QtWidgets.QPushButton("Save")
        self.fileOpsLayout.addWidget(self.Save_Button)

        # === Options and Threshold ===
        self.optionsLayout = QtWidgets.QVBoxLayout()
        self.groupBoxLayout.addWidget(QtWidgets.QLabel("Options & Threshold"))
        self.groupBoxLayout.addLayout(self.optionsLayout)

        self.grooming_limit_Label = QtWidgets.QLabel("Grooming Threshold:")
        self.grooming_limit_Label.setStyleSheet("color: white;")
        self.optionsLayout.addWidget(self.grooming_limit_Label)

        self.lineEdit_grooming_y = QtWidgets.QLineEdit()
        self.lineEdit_grooming_y.setFixedWidth(50)
        self.optionsLayout.addWidget(self.lineEdit_grooming_y)

        self.checkBox_face = QtWidgets.QCheckBox("Whisker Pad")
        self.optionsLayout.addWidget(self.checkBox_face)

        self.checkBox_pupil = QtWidgets.QCheckBox("Pupil")
        self.optionsLayout.addWidget(self.checkBox_pupil)

        self.checkBox_nwb = QtWidgets.QCheckBox("Save nwb")
        self.optionsLayout.addWidget(self.checkBox_nwb)

        self.save_video = QtWidgets.QCheckBox("Save Video")
        self.optionsLayout.addWidget(self.save_video)

        # Add the group box to the main layout
        self.mainLayout.addWidget(self.groupBox)

        # Set the main layout to the central widget
        self.centralwidget.setLayout(self.mainLayout)

    def setup_saturation(self):
        self.sliderLayout = QtWidgets.QVBoxLayout()
        self.saturation_Label = QtWidgets.QLabel("Saturation")
        self.saturation_Label.setAlignment(QtCore.Qt.AlignLeft)
        self.saturation_Label.setStyleSheet("color: white;")
        self.sliderLayout.addWidget(self.saturation_Label)
        self.saturation_slider_layout = QtWidgets.QHBoxLayout()
        self.saturation_Slider = functions.setup_sliders(self.centralwidget, 0, 100, 0, "horizontal")
        self.saturation_slider_layout.addWidget(self.saturation_Slider)
        self.lineEdit_satur_value = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_satur_value.setFixedWidth(50)
        self.saturation_slider_layout.addWidget(self.lineEdit_satur_value)
        self.sliderLayout.addLayout(self.saturation_slider_layout)
        self.Main_V_Layout.addLayout(self.sliderLayout)

    def setup_connections(self):
        self.LoadVideo.triggered.connect(self.load_handler.load_video)
        self.load_np.triggered.connect(self.load_handler.open_image_folder)
        self.saturation_Slider.valueChanged.connect(self.satur_value)
        self.Slider_frame.valueChanged.connect(self.Display_handler.update_frame_view)
        self.lineEdit_frame_number.editingFinished.connect(self.update_slider)
        self.Process_Button.clicked.connect(self.process_handler.process)
        self.Add_eyecorner.clicked.connect(self.eyecorner_clicked)
        self.Undo_blinking_Button.clicked.connect(self.init_undo_blinking)
        self.detect_blinking_Button.clicked.connect(self.start_blinking_detection)
        self.Save_Button.clicked.connect(self.save_handler.init_save_data)
        self.grooming_Button.clicked.connect(self.change_cursor_color)
        self.Undo_grooming_Button.clicked.connect(self.undo_grooming)
        self.Erase_Button.clicked.connect(self.graphicsView_subImage.activateEraseMode)
        self.Undo_Erase_Button.clicked.connect(self.graphicsView_subImage.undoBrushStrokes)



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


    def set_frame(self, face_frame=None, Pupil_frame=None, reflect_ellipse = None):
        if face_frame is not None:
            self.Face_frame = face_frame
        if Pupil_frame is not None:
            self.Pupil_frame = Pupil_frame
        if reflect_ellipse is not None:
            self.reflect_ellipse = reflect_ellipse



    def pupil_check(self):
        return self.checkBox_pupil.isChecked()

    def face_check(self):
        return self.checkBox_face.isChecked()
    def nwb_check(self):
        return self.checkBox_nwb.isChecked()
    def init_erasing_pixel(self):
        self.Eraser_active = True

    def satur_value(self, value):
        """Update saturation value and apply changes to the displayed sub-region."""

        # Update the text box with the new saturation value
        self.lineEdit_satur_value.setText(str(value))

        # Store the saturation value
        self.saturation = value

        # Ensure `sub_region` and `scene2` exist before trying to update the display
        if hasattr(self, 'sub_region') and self.sub_region is not None:
            if not hasattr(self, 'scene2') or self.scene2 is None:
                self.scene2 = QtWidgets.QGraphicsScene()  # Initialize if missing

            # Update the display with the new saturation
            functions.display_sub_region(
                self.graphicsView_subImage, self.sub_region, self.scene2, "pupil", self.saturation
            )

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
        pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner, width, height =\
            self.process_handler.pupil_dilation_comput(images, self.saturation, self.erased_pixels, self.reflect_ellipse)
        self.final_pupil_area = pupil_dilation
        self.X_saccade_updated = X_saccade
        self.Y_saccade_updated = Y_saccade
        return pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner,width, height



    def start_blinking_detection(self):
        if hasattr(self, 'pupil_dilation'):
            self.blinking_ids = self.process_handler.detect_blinking(self.pupil_dilation, self.width, self.height, self.X_saccade, self.Y_saccade)


        else:

            self.warning("Process Pupil first")



    def display_removed_grooming(self, grooming_thr, facemotion ):
        self.facemotion_without_grooming, self.grooming_ids, self.grooming_thr = self.process_handler.remove_grooming(grooming_thr, facemotion)
        print("grooming_ids", self.grooming_ids)
        self.plot_handler.plot_result(self.facemotion_without_grooming, self.graphicsView_whisker, "motion")

    def undo_grooming(self):
        if hasattr(self, 'motion_energy'):
            self.plot_handler.plot_result(self.motion_energy, self.graphicsView_whisker, "motion")

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