from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import traceback
from tqdm import tqdm
import cv2
class PupilWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, images,process_handler,saturation, contrast,erased_pixels,
                 brightness_concave_power, secondary_direction, reflect_ellipse,
                 mnd,reflect_brightness, clustering_method,binary_method,binary_threshold, saturation_method,saturation_ununiform
                 , primary_direction, brightness, brightness_curve,secondary_BrightGain,c_value,block_size, sub_image):
        super().__init__()
        self.images = images
        self.process_handler = process_handler
        self.saturation = saturation
        self.saturation_ununiform = saturation_ununiform
        self.contrast = contrast
        self.erased_pixels = erased_pixels
        self.reflect_ellipse = reflect_ellipse
        self.mnd = mnd
        self.reflect_brightness = reflect_brightness
        self.clustering_method = clustering_method
        self.saturation_method = saturation_method
        self.binary_method = binary_method
        self.binary_threshold = binary_threshold
        self.brightness = brightness
        self.brightness_curve = brightness_curve
        self.secondary_BrightGain = secondary_BrightGain
        self.brightness_concave_power = brightness_concave_power
        self.saturation_ununiform = saturation_ununiform
        self.primary_direction = primary_direction
        self.secondary_direction = secondary_direction
        self.c_value = c_value
        self.block_size = block_size
        self.sub_image = sub_image

    def run(self):


        try:
            result = self.process_handler.pupil_dilation_comput(
                self.images,self.saturation, self.contrast, self.erased_pixels,
                self.reflect_ellipse, self.mnd,self.reflect_brightness, self.clustering_method, self.binary_method,
                self.binary_threshold, self.saturation_method, self.brightness, self.brightness_curve,
                self.secondary_BrightGain, self.brightness_concave_power,
                self.saturation_ununiform, self.primary_direction, self.secondary_direction, self.c_value, self.block_size, self.sub_image)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            raise

class MotionWorker(QObject):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, images, face_frame):
        super().__init__()
        self.images = images
        self.face_frame = face_frame
        self._is_running = True

    def run(self):
        try:
            y1, y2, x1, x2 = self.face_frame

            # Case 1: 3D ndarray (n_frames, H, W) — vectorized fast path
            if isinstance(self.images, np.ndarray) and self.images.ndim == 3:
                roi_stack = self.images[:, y1:y2, x1:x2].astype(np.float32)

                diffs = np.diff(roi_stack, axis=0)
                motion_energy_values = np.mean(diffs ** 2, axis=(1, 2))
                motion_energy_values = np.insert(motion_energy_values, 0, motion_energy_values[0])

                # Emit 100% once
                if hasattr(self, "progress"):
                    self.progress.emit(100)

                self.finished.emit(motion_energy_values)
                return

            # Case 2: iterable/stream of frames — streaming path (no tqdm)
            previous_ROI = None
            motion_energy_values = []

            # progress control
            total = len(self.images) if hasattr(self.images, "__len__") else None
            done = 0
            last_bucket = -10  # so first bucket 0/10 can pass if desired

            for frame in self.images:
                if not self._is_running:
                    break

                roi = frame[y1:y2, x1:x2].astype(np.float32)
                if previous_ROI is not None:
                    diff = cv2.absdiff(roi, previous_ROI)
                    motion_energy_values.append(np.mean(diff * diff))
                previous_ROI = roi

                # progress every 10% if total is known
                done += 1
                if total:
                    pct = (done * 100) // total  # 0..100
                    bucket = (pct // 10) * 10  # 0,10,20,...,100
                    if bucket > last_bucket and hasattr(self, "progress"):
                        last_bucket = bucket
                        self.progress.emit(int(bucket))

            if motion_energy_values:
                motion_energy_values.insert(0, motion_energy_values[0])

            # If total was unknown, send 100% at the end
            if not total and hasattr(self, "progress"):
                self.progress.emit(100)

            self.finished.emit(np.array(motion_energy_values))

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False
