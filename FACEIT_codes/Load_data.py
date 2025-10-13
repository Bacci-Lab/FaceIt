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

import logging, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _fmt_secs(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

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

            t0 = time.perf_counter()
            logging.info(f"[Video] Opening: {self.app_instance.folder_path}")

            cap = cv2.VideoCapture(self.app_instance.folder_path)
            if not cap.isOpened():
                logging.error(f"[Video] Cannot open {self.app_instance.folder_path}")
                return
            self.app_instance.cap = cap

            # Probe props (cheap) — helpful in logs
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            logging.info(f"[Video] Probed props: {total_frames} frames, {fps_src:.2f} fps, {w}x{h}")

            self.app_instance.len_file = total_frames
            self.app_instance.Slider_frame.setMaximum(max(0, total_frames - 1))
            self.app_instance.video = True
            self.app_instance.NPY = False

            # First frame display/setup
            self.display_graphics(self.app_instance.folder_path)

            dt = time.perf_counter() - t0
            logging.info(f"✅ [Video] Prepared UI & metadata in {_fmt_secs(dt)}")

            # Enable UI
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
        t0 = time.perf_counter()
        file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        n = len(file_list)
        logging.info(f"[Load] Start loading {n} .npy images from {directory}")

        images = [None] * n  # Preallocate to maintain order
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_image, file, image_height): i
                for i, file in enumerate(file_list)
            }

            # Optional: tqdm (kept as-is). It won’t impact data.
            with tqdm(total=n, desc="Loading .npy images") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    image = future.result()
                    if image is not None:
                        images[index] = image
                    pbar.update(1)

        dt = time.perf_counter() - t0
        fps = (n / dt) if dt > 0 else float("inf")
        logging.info(f"✅ [Load] Images ready: {n} frames in {_fmt_secs(dt)} (~{fps:.1f} fps)")
        return images

    def load_frames_from_video(self, video_path, image_height, buffer_size=256):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"[Video] Cannot open {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_queue = queue.Queue(maxsize=buffer_size)

        t0 = time.perf_counter()
        produced = 0

        logging.info(f"[Stream] Start decoding {total_frames} frames from {video_path}")

        # progress bucket state for stream logging
        last_bucket = -1  # so first 0% can show if you want; keep -1 to start at 0->10 quickly

        def producer():
            nonlocal produced, last_bucket
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    aspect_ratio = w / h if h else 1.0
                    new_w = int(image_height * aspect_ratio)
                    frame_resized = cv2.resize(frame, (new_w, image_height), interpolation=cv2.INTER_AREA)
                    frame_queue.put(frame_resized)
                    produced += 1

                    # --- log every 10% instead of every 500 frames ---
                    if total_frames:
                        pct = int((produced * 100) // total_frames)
                        bucket = max(0, min(100, (pct // 10) * 10))
                        if bucket > last_bucket:
                            last_bucket = bucket
                            dt = time.perf_counter() - t0
                            fps = produced / dt if dt > 0 else 0.0
                            logging.info(f"[Stream] {bucket}% ({produced}/{total_frames}) "
                                         f"({_fmt_secs(dt)} @ {fps:.1f} fps)")
            finally:
                frame_queue.put(None)  # sentinel

        threading.Thread(target=producer, daemon=True).start()

        consumed = 0
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            consumed += 1
            yield frame

        cap.release()
        dt_all = time.perf_counter() - t0
        fps_all = produced / dt_all if dt_all > 0 else float("inf")
        logging.info(f"✅ [Stream] Finished. Produced={produced}, Consumed={consumed}, "
                     f"Total {_fmt_secs(dt_all)} (~{fps_all:.1f} fps)")

    def display_graphics(self, folder_path):
        """Display initial graphics and setup scenes."""
        t0 = time.perf_counter()
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
        dt = time.perf_counter() - t0
        logging.info(f"✅ [UI] First frame displayed in {_fmt_secs(dt)}")

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