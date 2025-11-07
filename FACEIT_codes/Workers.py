# FACEIT_codes/Workers.py
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
import traceback

class PupilWorker(QObject):
    """
    Runs pupil pipeline off the GUI thread.
    IMPORTANT: This worker must be GUI-free.
    """
    finished = pyqtSignal(object)   # tuple result consumed by handle_pupil_results
    error    = pyqtSignal(str)

    def __init__(
        self,
        images,
        process_handler,
        saturation,
        contrast,
        erased_pixels,
        brightness_concave_power,
        secondary_direction,
        reflect_ellipse,
        mnd,
        reflect_brightness,
        clustering_method,
        binary_method,
        binary_threshold,
        saturation_method,
        saturation_ununiform,
        primary_direction,
        brightness,
        brightness_curve,
        secondary_BrightGain,
        c_value,
        block_size,
        sub_image
    ):
        super().__init__()
        self.images = images
        self.process_handler = process_handler
        self.saturation = saturation
        self.contrast = contrast
        self.erased_pixels = erased_pixels
        self.brightness_concave_power = brightness_concave_power
        self.secondary_direction = secondary_direction
        self.reflect_ellipse = reflect_ellipse
        self.mnd = mnd
        self.reflect_brightness = reflect_brightness
        self.clustering_method = clustering_method
        self.binary_method = binary_method
        self.binary_threshold = binary_threshold
        self.saturation_method = saturation_method
        self.saturation_ununiform = saturation_ununiform
        self.primary_direction = primary_direction
        self.brightness = brightness
        self.brightness_curve = brightness_curve
        self.secondary_BrightGain = secondary_BrightGain
        self.c_value = c_value
        self.block_size = block_size
        self.sub_image = sub_image

    def run(self):
        try:
            result = self.process_handler.pupil_dilation_comput(
                self.images,
                self.saturation, self.contrast, self.erased_pixels,
                self.reflect_ellipse, self.mnd, self.reflect_brightness,
                self.clustering_method, self.binary_method, self.binary_threshold,
                self.saturation_method, self.brightness, self.brightness_curve,
                self.secondary_BrightGain, self.brightness_concave_power,
                self.saturation_ununiform, self.primary_direction, self.secondary_direction,
                self.c_value, self.block_size, self.sub_image
            )
            self.finished.emit(result)
        except Exception as e:
            # Never re-raise inside QThread; report via signal so debugger/Qt loop stays stable.
            tb = "".join(traceback.format_exc())
            self.error.emit(f"{type(e).__name__}: {e}\n{tb}")


class MotionWorker(QObject):
    """
    Computes whisker/face motion energy off the GUI thread.
    """
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error    = pyqtSignal(str)

    def __init__(self, images, face_frame):
        super().__init__()
        self.images = images
        self.face_frame = face_frame
        self._is_running = True

    def stop(self):
        self._is_running = False

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _normalize_box(face_frame):
        """
        face_frame is expected as (y0, y1, x0, x1) or similar floats.
        Returns ints (y0, y1, x0, x1), sorted so y0<y1, x0<x1, with y0/x0 >= 0.
        Raises ValueError if NaNs or wrong shape.
        """
        arr = np.asarray(face_frame, dtype=float).reshape(-1)
        if arr.shape != (4,) or not np.isfinite(arr).all():
            raise ValueError("Face ROI is not defined (contains NaN or wrong shape).")

        y0, y1, x0, x1 = arr
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0 = max(0, y0)
        x0 = max(0, x0)
        return y0, y1, x0, x1

    @staticmethod
    def _to_gray_stack(stack):
        """
        stack: (T,H,W) gray or (T,H,W,C) color.
        Returns float32 gray stack (T,H,W).
        """
        if stack.ndim == 3:
            return stack.astype(np.float32)
        if stack.ndim == 4:
            # Fast luminance-ish conversion without cv2 inside a loop
            if stack.shape[-1] == 3:
                # simple average; good enough for motion energy
                return stack.mean(axis=3, dtype=np.float32)
            if stack.shape[-1] == 4:
                return stack[..., :3].mean(axis=3, dtype=np.float32)
        raise ValueError(f"Unsupported image stack shape: {stack.shape}")

    # --- main ---------------------------------------------------------------
    def run(self):
        try:
            y0, y1, x0, x1 = self._normalize_box(self.face_frame)

            # Vectorized path for numpy stacks
            if isinstance(self.images, np.ndarray):
                if self.images.ndim not in (3, 4):
                    raise ValueError(f"Unsupported ndarray shape: {self.images.shape}")

                H = self.images.shape[1]
                W = self.images.shape[2]
                # clamp upper bounds to image size (positive overshoot is safe, negatives were clamped above)
                y1 = min(y1, H)
                x1 = min(x1, W)

                if y1 <= y0 or x1 <= x0:
                    raise ValueError("Face ROI has zero/negative size after normalization.")

                roi_stack = self.images[:, y0:y1, x0:x1]
                gray_stack = self._to_gray_stack(roi_stack)

                diffs = np.diff(gray_stack, axis=0)
                me = np.mean(diffs * diffs, axis=(1, 2)).astype(np.float32)
                # keep length == T
                if me.size > 0:
                    me = np.insert(me, 0, me[0]).astype(np.float32)
                else:
                    me = np.zeros((0,), dtype=np.float32)

                self.progress.emit(100)
                self.finished.emit(me)
                return

            # Streaming / iterable path
            prev = None
            values = []
            total = len(self.images) if hasattr(self.images, "__len__") else None
            done = 0
            last_bucket = -10

            for frame in self.images:
                if not self._is_running:
                    break

                # clamp high bounds lazily (frame-dependent W/H)
                H, W = frame.shape[:2]
                yy1 = min(y1, H)
                xx1 = min(x1, W)
                if yy1 <= y0 or xx1 <= x0:
                    raise ValueError("Face ROI has zero/negative size for current frame.")

                roi = frame[y0:yy1, x0:xx1]
                if roi.ndim == 3:
                    roi = roi.mean(axis=2, dtype=np.float32)
                else:
                    roi = roi.astype(np.float32)

                if prev is not None:
                    diff = cv2.absdiff(roi, prev)
                    values.append(float(np.mean(diff * diff)))
                prev = roi

                done += 1
                if total:
                    pct = (done * 100) // total
                    bucket = (pct // 10) * 10
                    if bucket > last_bucket:
                        last_bucket = bucket
                        self.progress.emit(int(bucket))

            if values:
                values.insert(0, values[0])
            elif total in (0, None):
                values = []

            if not total:
                self.progress.emit(100)

            self.finished.emit(np.array(values, dtype=np.float32))

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
