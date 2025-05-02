from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import traceback
class PupilWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, images,process_handler,saturation, contrast,erased_pixels,
                 brightness_concave_power, secondary_direction, reflect_ellipse,
                 mnd, clustering_method,binary_method,binary_threshold, saturation_method,saturation_ununiform
                 , primary_direction, brightness, brightness_curve,secondary_BrightGain, sub_image):
        super().__init__()
        self.images = images
        self.process_handler = process_handler
        self.saturation = saturation
        self.saturation_ununiform = saturation_ununiform
        self.contrast = contrast
        self.erased_pixels = erased_pixels
        self.reflect_ellipse = reflect_ellipse
        self.mnd = mnd
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
        self.sub_image =sub_image

    def run(self):


        try:
            result = self.process_handler.pupil_dilation_comput(
                self.images,self.saturation, self.contrast, self.erased_pixels,
                self.reflect_ellipse, self.mnd, self.clustering_method, self.binary_method,
                self.binary_threshold, self.saturation_method, self.brightness, self.brightness_curve,
                self.secondary_BrightGain, self.brightness_concave_power,
                self.saturation_ununiform, self.primary_direction, self.secondary_direction, self.sub_image)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()  # Show full error with line number
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
