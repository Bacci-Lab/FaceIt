import os
import numpy as np
import cv2
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PyQt5 import QtWidgets
from FACEIT_codes import functions
import threading
class LoadData:
    def __init__(self, app_instance):
        # Store a reference to the main app instance
        self.app_instance = app_instance
    def open_image_folder(self):
        """Open a folder dialog to select a directory containing .npy image files."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_path = os.path.join(project_root, "test_data", "test_images")

        # Open folder selection dialog
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.app_instance, "Select Folder", default_path)

        if not folder_path:
            return

        self.app_instance.folder_path = folder_path
        self.app_instance.save_path = folder_path
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]


        # Configure application settings if valid .npy files are found
        self.reset_GUI()
        self.app_instance.len_file = len(npy_files)
        self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
        self.app_instance.NPY = True
        self.app_instance.video = False
        self.display_graphics(folder_path)

        # Enable buttons after successful file check
        self.app_instance.FaceROIButton.setEnabled(True)
        self.app_instance.PlayPause_Button.setEnabled(True)
        self.app_instance.PupilROIButton.setEnabled(True)
        self.app_instance.Slider_frame.setEnabled(True)

    def load_video(self):
        """Load video and prepare for processing."""
        self.app_instance.folder_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.app_instance, "Load Video", "", "Video Files (*.avi *.wmv *mp4)"
        )
        if self.app_instance.folder_path:
            directory_path = os.path.dirname(self.app_instance.folder_path)
            self.app_instance.save_path = directory_path
            self.reset_GUI()
            self.app_instance.cap = cv2.VideoCapture(self.app_instance.folder_path)
            self.app_instance.len_file = int(self.app_instance.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
            self.app_instance.video = True
            self.app_instance.NPY = False
            self.display_graphics(self.app_instance.folder_path)
            self.app_instance.FaceROIButton.setEnabled(True)
            self.app_instance.PlayPause_Button.setEnabled(True)
            self.app_instance.PupilROIButton.setEnabled(True)
            self.app_instance.Slider_frame.setEnabled(True)

    def load_image(self, filepath, image_height):
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

    def load_images_from_directory(self, directory, image_height, max_workers=8):
        """Load images from a directory using multithreading while preserving order and improving performance."""
        start_time = time.time()
        file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        images = [None] * len(file_list)  # Preallocate the list to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_image, file, image_height): i
                for i, file in enumerate(file_list)
            }

            with tqdm(total=len(file_list), desc="Loading .npy images") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    image = future.result()
                    if image is not None:
                        images[index] = image
                    pbar.update(1)

        elapsed_time = time.time() - start_time
        print(f"✅ Image loading completed in {elapsed_time:.2f} seconds.")
        return images

    def load_frames_from_video(self, video_path, image_height, buffer_size=64):
        """Stream frames from a video file using a background thread and buffer (fast + memory-safe)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_queue = queue.Queue(maxsize=buffer_size)

        def producer():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # ✅ resize here once
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                new_w = int(image_height * aspect_ratio)
                frame_resized = cv2.resize(frame, (new_w, image_height), interpolation=cv2.INTER_AREA)
                frame_queue.put(frame_resized)
            frame_queue.put(None)  # sentinel

        # Start producer in background
        threading.Thread(target=producer, daemon=True).start()

        # Consumer yields frames one by one
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            yield frame

        cap.release()

    def display_graphics(self, folder_path):
        """Display initial graphics and setup scenes."""
        self.app_instance.frame = 0
        if self.app_instance.NPY:
            self.app_instance.image = functions.load_npy_by_index(folder_path, self.app_instance.frame)
        elif self.app_instance.video:
            self.app_instance.image = functions.load_frame_by_index(self.app_instance.cap, self.app_instance.frame)

        functions.initialize_attributes(self.app_instance, self.app_instance.image)
        self.app_instance.scene2 = functions.second_region(
            self.app_instance.graphicsView_subImage,
            self.app_instance.graphicsView_MainFig,
        )
        self.app_instance.graphicsView_MainFig, self.app_instance.scene = functions.display_region(
            self.app_instance.image,
            self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width,
            self.app_instance.image_height
        )

    def reset_GUI(self):
        app = self.app_instance

        # === Reset internal states ===
        app.Eraser_active = False
        app.find_grooming_threshold = False
        app.contrast = 1
        app.brightness = 1
        app.brightness_curve = 1
        app.brightness_concave_power = 1.5
        app.saturation = 0
        app.saturation_ununiform = 1
        app.Show_binary = False
        app.primary_direction = None
        app.secondary_direction = None
        app.saturation_method = "None"
        app.clustering_method = "SimpleContour"
        app.binary_method = "Adaptive"
        app.mnd = 3
        app.c_value = 5
        app.block_size = 17
        app.reflect_brightness = 985
        app.binary_threshold = 220
        app.frame = 0
        app.Image_loaded = False
        app.Pupil_ROI_exist = False
        app.Face_ROI_exist = False
        app.eye_corner_mode = False
        app.eyecorner = None
        app.eye_corner_center = None
        app.erased_pixels = None

        # === Reset GUI elements ===

        # Sliders and lineEdits
        app.saturation_Slider.setValue(0)
        app.saturation_ununiform_Slider.setValue(10)
        app.contrast_Slider.setValue(10)
        app.BrightGain_primary_Slider.setValue(10)
        app.brightness_curve_Slider.setValue(10)
        app.brightGain_secondary_Slider.setValue(10)
        app.Slider_frame.setValue(0)
        app.Slider_frame.setEnabled(False)

        app.lineEdit_frame_number.setText("0")
        app.lineEdit_brightGain_primary_value.setText("1")
        app.lineEdit_brightness_curve_value.setText("1")
        app.lineEdit_brightGain_secondary_value.setText("1")
        app.lineEdit_brightness_concave_power.setText("1")
        app.lineEdit_contrast_value.setText("1")
        app.lineEdit_satur_value.setText("0")
        app.lineEdit_satur_ununiform_value.setText("1")
        app.lineEdit_grooming_y.setText("0")

        # Checkboxes
        app.checkBox_binary.setChecked(False)
        app.checkBox_pupil.setChecked(False)
        app.checkBox_face.setChecked(False)
        app.checkBox_nwb.setChecked(False)
        app.save_video.setChecked(False)

        # Buttons
        app.Add_eyecorner.setEnabled(False)
        app.ReflectionButton.setEnabled(False)
        app.Erase_Button.setEnabled(False)
        app.Process_Button.setEnabled(False)
        app.Save_Button.setEnabled(False)
        app.checkBox_face.setEnabled(False)
        app.checkBox_pupil.setEnabled(False)

        # Radio buttons
        app.radioButton_SimpleContour.setChecked(True)
        app.radio_button_none.setChecked(True)
        app.radio_button_none_secondary.setChecked(True)
        #-------------------
        app.adjustment_mode_group.removeButton(app.radio_button_Uniform)
        app.adjustment_mode_group.removeButton(app.radio_button_Gradual)

        # 2. Uncheck both manually
        app.radio_button_Uniform.setChecked(False)
        app.radio_button_Gradual.setChecked(False)

        # 3. Add them back (optional)
        app.adjustment_mode_group.addButton(app.radio_button_Uniform)
        app.adjustment_mode_group.addButton(app.radio_button_Gradual)
        #--------------------------------------------