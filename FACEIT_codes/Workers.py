from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

class PupilWorker(QObject):
    finished = pyqtSignal(object)  # signal to send results back
    error = pyqtSignal(str)

    def __init__(self, images, process_handler, saturation, contrast, erased_pixels,
                 reflect_ellipse, mnd, binary_threshold, clustering_method, saturation_method,saturation_ununiform):
        super().__init__()
        self.images = images
        self.process_handler = process_handler
        self.saturation = saturation
        self.saturation_ununiform = saturation_ununiform
        self.contrast = contrast
        self.erased_pixels = erased_pixels
        self.reflect_ellipse = reflect_ellipse
        self.mnd = mnd
        self.binary_threshold = binary_threshold
        self.clustering_method = clustering_method
        self.saturation_method = saturation_method

    def run(self):

        try:
            result = self.process_handler.pupil_dilation_comput(
                self.images, self.saturation,self.saturation_ununiform, self.contrast, self.erased_pixels,
                self.reflect_ellipse, self.mnd, self.binary_threshold, self.clustering_method
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MotionWorker(QObject):
    finished = pyqtSignal(np.ndarray)  # Emit the result
    progress = pyqtSignal(int)         # Optional: for updating progress bar
    error = pyqtSignal(str)            # NEW: emit in case of exceptions

    def __init__(self, images, face_frame):
        super().__init__()
        self.images = images
        self.face_frame = face_frame
        self._is_running = True

    def run(self):
        try:
            previous_ROI = None
            motion_energy_values = []
            total = len(self.images)

            for i, current_array in enumerate(self.images):
                if not self._is_running:
                    return

                current_ROI = current_array[
                    self.face_frame[0]:self.face_frame[1],
                    self.face_frame[2]:self.face_frame[3]
                ].flatten()

                if previous_ROI is not None:
                    motion_energy_value = np.mean((current_ROI - previous_ROI) ** 2)
                    motion_energy_values.append(motion_energy_value)

                previous_ROI = current_ROI
                self.progress.emit(i + 1)

            if motion_energy_values:
                motion_energy_values.insert(0, motion_energy_values[0])

            self.finished.emit(np.array(motion_energy_values))

        except Exception as e:
            self.error.emit(str(e))  # Handle unexpected exceptions

    def stop(self):
        self._is_running = False
