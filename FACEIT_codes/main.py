from PyQt5.QtWidgets import QMessageBox
from FACEIT_codes import functions
import numpy as np
from analysis import ProcessHandler
from Save import SaveHandler
from Load_data import LoadData
from Graphical_ROIS import ROIHandler
from FACEIT_codes import display_and_plots
from PyQt5 import QtWidgets, QtCore, QtGui
from GUI_Intractions import  GUI_Intract

class FaceMotionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the class that has the `process` function
        self.process_handler = ProcessHandler(self)
        self.save_handler = SaveHandler(self)
        self.load_handler = LoadData(self)
        self.plot_handler = display_and_plots.PlotHandler(self)
        self.Display_handler = display_and_plots.Display(self)

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
        self.roi_handler = ROIHandler(self)
        MainWindow.showMaximized()
        self.PupilROIButton.clicked.connect(lambda: self.execute_pupil_roi() if self.NPY or self.video else self.warning("Load data to analyse"))
        self.FaceROIButton.clicked.connect(lambda: self.execute_face_roi() if self.NPY or self.video else self.warning("Load data to analyse"))
        self.ReflectionButton.clicked.connect(lambda: self.execute_reflect_roi())
        self.Add_blank_button.clicked.connect(lambda: self.execute_blank_roi())

    def execute_reflect_roi(self):
        # Call `add_roi` to display a 'reflection' ROI
        self.roi_handler.Add_ROI(
            roi_type='reflection',
            roi_center=self.reflection_center,
            image=self.image,
            height=self.reflect_height,
            width=self.reflect_width,
            color='gray',
            handle_size=3
        )
    def execute_blank_roi(self):
        self.roi_handler.Add_ROI(
            roi_type='blank',
            roi_center=self.blank_R_center,
            image=self.image,
            height= self.blank_height,
            width = self.blank_width,
            color='black',
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
            Button2=self.Add_blank_button,
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
        self.graphicsView_MainFig = GUI_Intract(self.centralwidget)
        self.graphicsView_MainFig.parent = self
        self.Image_H_Layout.addWidget(self.graphicsView_MainFig)
        self.graphicsView_subImage = GUI_Intract(self.centralwidget)
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






    def satur_value(self, value):
        self.lineEdit_satur_value.setText(str(self.saturation_Slider.value()))
        self.saturation = value
        if self.sub_region is not None:
            _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.scene2,
                                                 "pupil", self.saturation)
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


    def start_pupil_dilation_computation(self, images):
        pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner, width, height =\
            self.process_handler.pupil_dilation_comput(images, self.saturation,self.blank_ellipse, self.reflect_ellipse)
        self.final_pupil_area = pupil_dilation
        self.X_saccade_updated = X_saccade
        self.Y_saccade_updated = Y_saccade
        return pupil_dilation, pupil_center_X, pupil_center_y,pupil_center,\
            X_saccade, Y_saccade, pupil_distance_from_corner,width, height



    def start_blinking_detection(self):
        if hasattr(self, 'pupil_dilation'):
            self.process_handler.detect_blinking(self.pupil_dilation, self.width, self.height, self.X_saccade, self.Y_saccade)

        else:
            print("self.pupil_dilation does not exist")
            self.warning("Process Pupil first")


    def remove_grooming(self,grooming_thr, facemotion):
        grooming_ids = np.where(facemotion>=grooming_thr)
        facemotion = np.array(facemotion)
        self.facemotion_without_grooming = np.copy(facemotion)
        self.facemotion_without_grooming[grooming_ids] = grooming_thr
        return self.facemotion_without_grooming

    def display_removed_grooming(self, grooming_thr, facemotion ):
        print(" grooming_thr is ", grooming_thr )
        self.facemotion_without_grooming = self.remove_grooming(grooming_thr, facemotion)
        self.plot_handler.plot_result(self.facemotion_without_grooming, self.graphicsView_whisker, "motion")

    def undo_grooming(self):
        self.plot_handler.plot_result(self.motion_energy, self.graphicsView_whisker, "motion")

    def init_undo_blinking(self):
        if hasattr(self, 'pupil_dilation'):
            self.Undo_blinking()

        else:
            self.warning("Process Pupil first")

    def Undo_blinking(self):
        self.final_pupil_area = np.array(self.pupil_dilation)
        self.X_saccade_updated = np.array(self.X_saccade)
        self.Y_saccade_updated = np.array(self.Y_saccade)
        self.plot_handler.plot_result(self.pupil_dilation, self.graphicsView_pupil, "pupil", color="palegreen",
                         saccade=self.X_saccade)

    def eyecorner_clicked(self):
        self.eye_corner_mode = True
        print("is true", self.eye_corner_mode )



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
