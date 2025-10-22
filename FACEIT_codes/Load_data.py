import os
import numpy as np
import cv2
import queue
from PyQt5.QtCore import QSignalBlocker
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
        self.stream_stop = threading.Event()

    def _resolve_video_path(self, explicit_path=None):
        """Return a valid video path string or None."""
        path = explicit_path if explicit_path else getattr(self.app_instance, "video_path", None)
        if isinstance(path, str) and path.strip():
            return path
        logging.error("[Video] Cannot open None")
        return None


    def open_image_folder(self):
        """Open a folder dialog to select a directory containing .npy image files."""
        self.reset_GUI()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_path = os.path.join(project_root, "test_data", "test_images")

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.app_instance, "Select Folder", default_path
        )
        if not folder_path:
            return

        # --- Cancel any background stream and reset UI ---
        if hasattr(self, "stream_stop") and self.stream_stop is not None:
            self.stream_stop.set()
            import threading
            self.stream_stop = threading.Event()



        # --- Switch to NPY mode, clear video state ---
        app = self.app_instance
        app.NPY = True
        app.video = False
        app.folder_path = folder_path
        app.video_path = None
        app.save_path = folder_path

        # Release any previous capture cleanly
        try:
            if getattr(app, "cap", None):
                app.cap.release()
        except Exception:
            pass
        app.cap = None

        # --- List .npy files (natural sort) and handle empty/invalid folder ---
        try:
            npy_files = functions.list_npy_files(folder_path)  # raises if invalid/empty
        except Exception as e:
            if hasattr(app, "warning"):
                app.warning(str(e))
            return  # keep slider disabled; nothing to load

        # Optional cache for faster per-frame loads
        app.npy_cache = {"dir": folder_path, "files": npy_files}
        app.npy_files = npy_files

        total = len(npy_files)
        app.len_file = total

        # --- Bound slider safely & sync the line edit ---
        blocker = QSignalBlocker(app.Slider_frame)
        app.Slider_frame.setMinimum(0)
        app.Slider_frame.setMaximum(max(0, total - 1))
        app.Slider_frame.setValue(0)
        del blocker
        app.lineEdit_frame_number.setText("0")
        if hasattr(app, "update_frame_validator"):
            app.update_frame_validator()

        # --- First frame display/setup ---
        self.display_graphics(None)

        # --- Enable UI after success ---
        app.FaceROIButton.setEnabled(True)
        app.PlayPause_Button.setEnabled(True)
        app.PupilROIButton.setEnabled(True)
        app.Slider_frame.setEnabled(True)

    def load_video(self):
        self.reset_GUI()

        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.app_instance, "Load Video", "", "Video Files (*.avi *.wmv *.mp4)"
        )
        if not selected_path:
            return

        # --- Cancel any background stream and reset UI ---
        if hasattr(self, "stream_stop") and self.stream_stop is not None:
            self.stream_stop.set()
            import threading
            self.stream_stop = threading.Event()

        # --- VIDEO mode ON, NPY mode OFF ---
        app = self.app_instance
        app.video = True
        app.NPY = False

        # keep paths separate
        app.video_path = selected_path
        app.folder_path = None  # IMPORTANT: clear old NPY dir
        app.save_path = os.path.dirname(selected_path)

        # clear any old NPY cache
        app.npy_cache = None
        app.npy_files = None

        # close previous capture if any
        try:
            if getattr(app, "cap", None):
                app.cap.release()
        except Exception:
            pass

        t0 = time.perf_counter()
        logging.info(f"[Video] Opening: {selected_path}")
        cap = cv2.VideoCapture(selected_path)
        if not cap.isOpened():
            logging.error(f"[Video] Cannot open {selected_path}")
            return
        app.cap = cap

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        logging.info(f"[Video] Probed props: {total_frames} frames, {fps_src:.2f} fps, {w}x{h}")

        if total_frames <= 0:
            logging.error(f"[Video] No frames in {selected_path}")
            cap.release()
            app.cap = None
            return

        app.len_file = total_frames

        # bound the slider safely without emitting signals
        blocker = QSignalBlocker(app.Slider_frame)
        app.Slider_frame.setMinimum(0)
        app.Slider_frame.setMaximum(max(0, total_frames - 1))
        app.Slider_frame.setValue(0)
        del blocker
        app.lineEdit_frame_number.setText("0")
        if hasattr(app, "update_frame_validator"):
            app.update_frame_validator()

        # First frame display/setup
        self.display_graphics(None)

        dt = time.perf_counter() - t0
        logging.info(f"✅ [Video] Prepared UI & metadata in {_fmt_secs(dt)}")

        # Enable UI
        app.FaceROIButton.setEnabled(True)
        app.PlayPause_Button.setEnabled(True)
        app.PupilROIButton.setEnabled(True)
        app.Slider_frame.setEnabled(True)

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

    def load_frames_from_video(self, video_path=None, image_height=480, buffer_size=256, stop_event=None):
        """
        Stream frames from a video in a background thread.

        - If video_path is None, uses self.app_instance.video_path.
        - Always opens its own cv2.VideoCapture (no racing with UI 'cap').
        - Seeks to frame 0 before reading.
        - If stop_event (threading.Event) is set during streaming, exits cleanly.
        """
        # ---- Resolve path safely ----
        if video_path is None:
            video_path = getattr(self.app_instance, "video_path", None)
        if not isinstance(video_path, str) or not video_path.strip():
            logging.error("[Stream] Cannot open None")
            return

        # ---- Open a private capture and seek to 0 ----
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"[Stream] Cannot open {video_path}")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ---- Basic guards ----
        if not image_height or image_height <= 0:
            logging.error(f"[Stream] Invalid image_height={image_height}")
            cap.release()
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_queue = queue.Queue(maxsize=max(2, buffer_size))

        t0 = time.perf_counter()
        produced = 0
        last_bucket = -1
        logging.info(f"[Stream] Start decoding {total_frames} frames from {video_path}")

        def producer():
            nonlocal produced, last_bucket
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    h, w = frame.shape[:2]
                    if h <= 0 or w <= 0:
                        continue

                    aspect = w / float(h)
                    new_w = max(1, int(image_height * aspect))
                    frame_resized = cv2.resize(frame, (new_w, image_height), interpolation=cv2.INTER_AREA)

                    # avoid deadlock if consumer stalls
                    try:
                        frame_queue.put(frame_resized, timeout=0.5)
                    except queue.Full:
                        if stop_event is not None and stop_event.is_set():
                            break
                        continue

                    produced += 1

                    # progress log every 10%
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
                # always signal consumer to stop and release cap
                try:
                    frame_queue.put_nowait(None)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        pass
                    frame_queue.put_nowait(None)
                cap.release()

        threading.Thread(target=producer, daemon=True).start()

        consumed = 0
        while True:
            fr = frame_queue.get()
            if fr is None:
                break
            consumed += 1
            yield fr

        dt_all = time.perf_counter() - t0
        fps_all = produced / dt_all if dt_all > 0 else float("inf")
        logging.info(f"✅ [Stream] Finished. Produced={produced}, Consumed={consumed}, "
                     f"Total {_fmt_secs(dt_all)} (~{fps_all:.1f} fps)")

    def display_graphics(self, _unused):
        """Display initial graphics and setup scenes."""
        t0 = time.perf_counter()
        self.app_instance.frame = 0

        if self.app_instance.NPY:
            if not (self.app_instance.folder_path and os.path.isdir(self.app_instance.folder_path)):
                logging.error("[UI] NPY mode but folder_path is not a directory")
                return
            self.app_instance.image = functions.load_npy_by_index(self.app_instance.folder_path, 0)

        elif self.app_instance.video:
            if not getattr(self.app_instance, "cap", None):
                logging.error("[UI] Video mode but cap is None")
                return
            self.app_instance.image = functions.load_frame_by_index(self.app_instance.cap, 0)

        else:
            logging.error("[UI] No source loaded")
            return

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