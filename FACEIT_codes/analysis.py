
from PyQt5.QtCore import  QThread
from FACEIT_codes import pupil_detection
from FACEIT_codes.Workers import MotionWorker
from multiprocessing import Pool
import numpy as np
import math
import cv2
import time
from multiprocessing import cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from FACEIT_codes.functions import (
    change_saturation_uniform,
    change_Gradual_saturation,
    apply_intensity_gradient_gray,
    SaturationSettings, show_ROI2
)

def process_single_frame(args):
    (
        current_image, sub_image,
        saturation, contrast,
        erased_pixels, reflect_ellipse,
        eye_corner_center, mnd,
        reflect_brightness,
        clustering_method, binary_method,binary_threshold,
        saturation_method,
        brightness, brightness_curve,
        secondary_BrightGain, brightness_concave_power,
        saturation_ununiform,
        primary_direction, secondary_direction
    ) = args


    sub_region, _ = show_ROI2(sub_image, current_image)

    # === Build saturation settings ===
    settings = SaturationSettings(
        primary_direction=primary_direction,
        brightness_curve=brightness_curve,
        brightness=brightness,
        secondary_direction=secondary_direction,
        brightness_concave_power=brightness_concave_power,
        secondary_BrightGain = secondary_BrightGain,
        saturation_ununiform=saturation_ununiform
    )

    # === Apply saturation method ===
    if saturation_method == "Gradual":
        if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
            # Grayscale image
            processed = apply_intensity_gradient_gray(sub_region, settings)
        else:
            # Fake grayscale check
            if np.allclose(sub_region[..., 0], sub_region[..., 1]) and np.allclose(sub_region[..., 1], sub_region[..., 2]):
                gray = cv2.cvtColor(sub_region, cv2.COLOR_BGR2GRAY)
                processed = apply_intensity_gradient_gray(gray, settings)
            else:
                processed = change_Gradual_saturation(sub_region, settings)
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

    elif saturation_method == "Uniform":
        if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
            sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)
        processed = change_saturation_uniform(sub_region, saturation=saturation, contrast=contrast)
    else:
        processed = sub_region.copy()

    # === Convert to RGBA for pupil detection ===
    sub_region_rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)

    _, center, width, height, angle, current_area = pupil_detection.detect_pupil(
        sub_region_rgba, erased_pixels, reflect_ellipse, mnd,reflect_brightness, clustering_method, binary_method, binary_threshold
    )

    pupil_distance_from_corner = (
        math.sqrt((center[0] - eye_corner_center[0]) ** 2 + (center[1] - eye_corner_center[1]) ** 2)
        if eye_corner_center is not None else np.nan
    )

    center = (int(center[0]), int(center[1]))
    width, height = int(width), int(height)

    return current_area, center, center[0], center[1], width, height, pupil_distance_from_corner



def display_show_ROI(ROI, image):

    if isinstance(ROI, tuple):
        x, y, width, height = ROI
        sub_image = image[int(y):int(y + height), int(x):int(x + width)]
        return sub_image, ROI
    sub_image = ROI.rect()
    return sub_image, ROI


def create_roi(x, y, width, height):
    """Creates a representation of a region of interest (ROI) based on given coordinates and size."""
    # Return a simple dictionary or tuple with ROI properties
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height
    }

class ProcessHandler:
    def __init__(self, app_instance):
        # Store a reference to the app instance for accessing its attributes and methods
        self.app_instance = app_instance


    def process(self):
        """Processes pupil and face data if selected and images are loaded."""
        # Process pupil data if the pupil checkbox is checked
        if self.app_instance.pupil_check():
            self.process_pupil_data()

        # Process face data if the face checkbox is checked
        if self.app_instance.face_check():
            self.process_face_data()

    def process_pupil_data(self):
        """Processes pupil data asynchronously using a worker thread."""
        if not self.app_instance.Pupil_ROI_exist:
            self.app_instance.warning("NO Pupil ROI is chosen!")
            return

        self.load_images_if_needed()
        self.app_instance.start_pupil_dilation_computation(self.app_instance.images)

    def process_face_data(self):
        if not self.app_instance.Face_ROI_exist:
            self.app_instance.warning("NO Face ROI is chosen!")
            return

        self.load_images_if_needed()

        # Setup threading
        self.motion_thread = QThread()
        self.motion_worker = MotionWorker(self.app_instance.images, self.app_instance.Face_frame)

        self.motion_worker.moveToThread(self.motion_thread)

        # Signals
        self.motion_thread.started.connect(self.motion_worker.run)
        self.motion_worker.finished.connect(self.handle_motion_result)
        self.motion_worker.error.connect(self.handle_worker_error)


        # Cleanup
        self.motion_worker.finished.connect(self.motion_thread.quit)
        self.motion_worker.finished.connect(self.motion_worker.deleteLater)
        self.motion_thread.finished.connect(self.motion_thread.deleteLater)
        self.motion_thread.start()

    def handle_motion_result(self, result):
        self.app_instance.motion_energy = result
        self.app_instance.plot_handler.plot_result(
            result,
            self.app_instance.graphicsView_whisker,
            "motion"
        )

    def handle_worker_error(self, error_msg):
        self.app_instance.warning(f"Motion processing error: {error_msg}")

    def load_images_if_needed(self):
        """Loads images if they are not already loaded."""
        if not self.app_instance.Image_loaded:
            if self.app_instance.NPY:
                # Load images from a directory of .npy files
                self.app_instance.images = self.app_instance.load_handler.load_images_from_directory(self.app_instance.folder_path,self.app_instance.image_height)
            elif self.app_instance.video:
                # Load images from a video file
                self.app_instance.images = self.app_instance.load_handler.load_frames_from_video(self.app_instance.folder_path, self.app_instance.image_height)

            # Mark images as loaded
            self.app_instance.Image_loaded = True

    def detect_blinking(self,
                        pupil,
                        width,
                        height,
                        x_saccade,
                        y_saccade,
                        win=15,
                        k=3.0):
        """
        Detect blinks by flagging outliers in the pupil signal via a Hampel filter
        (rolling median ± k * scaled‐MAD), then mask and interpolate saccade data.

        Parameters:
            pupil      (array): 1D pupil-area (or diameter) time series
            width      (array): [unused]—kept for signature compatibility
            height     (array): [unused]—kept for signature compatibility
            x_saccade  (array): raw X‐saccade time series
            y_saccade  (array): raw Y‐saccade time series
            win        (int): half‐window length for rolling stats (default 25)
            k          (float): outlier threshold multiplier (default 3.0)
        Returns:
            combined_blinking_ids (list): indices of samples flagged as blinks
        """

        # 1) Copy inputs into instance attributes
        pupil = np.asarray(pupil)
        self.app_instance.X_saccade_updated = np.asarray(x_saccade)
        self.app_instance.Y_saccade_updated = np.asarray(y_saccade)

        # 2) Set up Hampel parameters
        L = 1.4826  # consistency factor → MAD ≈ σ for Gaussians
        n = len(pupil)
        medians = np.zeros(n)
        mads = np.zeros(n)
        outliers = np.zeros(n, dtype=bool)

        # 3) Rolling‐window Hampel filter
        half = win // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)  # +1 because Python slicing is exclusive
            window = pupil[start:end]

            med = np.median(window)
            mad = L * np.median(np.abs(window - med))

            medians[i] = med
            mads[i] = mad

            # Flag as blink if it exceeds k × local MAD
            if mad > 0 and np.abs(pupil[i] - med) > k * mad:
                outliers[i] = True

        # 4) Extract the blink indices
        blinking_ids = np.where(outliers)[0].tolist()

        # 5) Mask saccade samples at those times
        max_idx = self.app_instance.X_saccade_updated.shape[-1]
        valid_ids = [i for i in blinking_ids if i < max_idx]
        valid_ids.sort()

        self.app_instance.X_saccade_updated[0, valid_ids] = np.nan
        self.app_instance.Y_saccade_updated[0, valid_ids] = np.nan

        # 6) Interpolate pupil through blinks
        self.app_instance.interpolated_pupil = pupil_detection.interpolate(valid_ids, pupil)

        # 7) Plot final result
        self.app_instance.plot_handler.plot_result(
            self.app_instance.interpolated_pupil,
            self.app_instance.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.app_instance.X_saccade
        )

        # 8) Store and return
        self.app_instance.final_pupil_area = np.array(self.app_instance.interpolated_pupil)
        return valid_ids

    def pupil_dilation_comput(self, images, saturation, contrast, erased_pixels, reflect_ellipse,
                              mnd,reflect_brightness, clustering_method, binary_method,binary_threshold,
                              saturation_method, brightness, brightness_curve,
                              secondary_BrightGain, brightness_concave_power,
                              saturation_ununiform, primary_direction, secondary_direction, sub_image):
        """
        Computes pupil dilation and related metrics from a series of images using parallel processing.
        """
        start_time = time.time()

        eye_corner_center = (
            self.app_instance.eye_corner_center[0],
            self.app_instance.eye_corner_center[1]
        ) if self.app_instance.eye_corner_center is not None else None

        # Build frame_args for parallel processing
        frame_args = [
            (
                current_image, sub_image,
                saturation, contrast,
                erased_pixels, reflect_ellipse,
                eye_corner_center, mnd,reflect_brightness,
                clustering_method, binary_method,binary_threshold,
                saturation_method, brightness, brightness_curve,
                secondary_BrightGain, brightness_concave_power,
                saturation_ununiform, primary_direction, secondary_direction
            )
            for current_image in images
        ]

        num_workers = max(1, cpu_count() // 2)
        print(f"Using {num_workers} workers out of {cpu_count()} cores.")

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_frame, frame_args), total=len(frame_args)))

        pupil_dilation, pupil_center, pupil_center_X, pupil_center_y, pupil_width, pupil_height, pupil_distance_from_corner = zip(
            *results)

        elapsed_time = time.time() - start_time
        print(f"Time taken for pupil dilation computation: {elapsed_time:.2f} seconds")

        # Convert to numpy arrays
        pupil_dilation = np.array(pupil_dilation)
        pupil_center = np.array(pupil_center)
        pupil_center_X = np.array(pupil_center_X)
        pupil_center_y = np.array(pupil_center_y)
        pupil_width = np.array(pupil_width)
        pupil_height = np.array(pupil_height)
        pupil_distance_from_corner = np.array(pupil_distance_from_corner)

        X_saccade = self.Saccade(pupil_center_X)
        Y_saccade = self.Saccade(pupil_center_y)

        return (pupil_dilation, pupil_center_X, pupil_center_y, pupil_center,
                X_saccade, Y_saccade, pupil_distance_from_corner, pupil_width, pupil_height)




    def Saccade(self, pupil_center_i):
        """
        Computes the saccade movements based on changes in the pupil center coordinates.

        Parameters:
            pupil_center_i (array-like): The center coordinates of the pupil for each frame in the i axis.

        Returns:
            np.ndarray: A 2D array with saccade values, where small changes (<2) are replaced with NaN.
        """
        # Compute saccade differences using numpy's vectorized operations
        saccade = np.diff(pupil_center_i).astype(float)

        # Duplicate the first computed value at the beginning of the list to avoid changing dataset length
        if saccade.size > 0:
            first_value = saccade[0]
            saccade = np.insert(saccade, 0, first_value)


        # Set small movements (absolute value < 2) to NaN for better filtering of significant movements
        saccade[abs(saccade) < 2] = np.nan


        # Reshape the saccade array to be 2D (1 row) for consistency
        saccade = saccade.reshape(1, -1)


        #self.plot_saccade(pupil_center_i, saccade)



        return saccade

    def remove_grooming(self, grooming_thr, facemotion):
        """
        Caps the 'facemotion' data at a specified grooming threshold.

        Parameters:
        grooming_thr (float): The threshold above which 'facemotion' values are capped.
        facemotion (array-like): The array of facial motion data to be processed.

        Returns:
        tuple: A tuple containing:
            - facemotion_without_grooming (np.ndarray): The 'facemotion' array with values capped at the threshold.
            - grooming_ids (np.ndarray): Indices of elements in 'facemotion' that were capped.
            - grooming_thr (float): The grooming threshold used for capping.
        """
        # Ensure 'facemotion' is converted to a NumPy array for array operations.
        facemotion = np.asarray(facemotion)

        # Identify indices where 'facemotion' exceeds the grooming threshold.
        grooming_ids = np.where(facemotion >= grooming_thr)

        # Cap the 'facemotion' values at the threshold.
        facemotion_without_grooming = np.minimum(facemotion, grooming_thr)

        # Store the modified array as an attribute in the app instance.
        self.app_instance.facemotion_without_grooming = facemotion_without_grooming

        # Return the modified array, indices of capped values, and the threshold.
        return facemotion_without_grooming, grooming_ids, grooming_thr




