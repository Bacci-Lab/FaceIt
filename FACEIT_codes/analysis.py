
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from FACEIT_codes import pupil_detection, functions
from FACEIT_codes import display_and_plots
from FACEIT_codes.Workers import MotionWorker
from multiprocessing import Pool
import numpy as np
import math
import cv2
import time
from multiprocessing import cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from FACEIT_codes.functions import change_saturation_uniform


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

def process_single_frame(args):
    current_image, frame, saturation, contrast, erased_pixels, reflect_ellipse, eye_corner_center, mnd, binary_threshold, clustering_method = args
    sub_region = current_image[frame[0]:frame[1], frame[2]:frame[3]]

    if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
        sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)

    # Now correctly call the utility function with saturation and contrast:
    sub_region = change_saturation_uniform(sub_region, saturation=saturation, contrast=contrast)

    sub_region_rgba = cv2.cvtColor(sub_region, cv2.COLOR_BGR2BGRA)

    _, center, width, height, angle, current_area = pupil_detection.detect_pupil(
        sub_region_rgba, erased_pixels, reflect_ellipse, mnd, binary_threshold, clustering_method
    )


    pupil_distance_from_corner = (
        math.sqrt((center[0] - eye_corner_center[0]) ** 2 + (center[1] - eye_corner_center[1]) ** 2)
        if eye_corner_center is not None else np.nan
    )

    center = (int(center[0]), int(center[1]))
    width, height = int(width), int(height)

    return current_area, center, center[0], center[1], width, height, pupil_distance_from_corner


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

    def detect_blinking(self, pupil, width, height, x_saccade, y_saccade):
        """
        Detects and processes blinking events based on pupil data, width, and height ratios.

        Parameters:
            pupil (array): Array of pupil data.
            width (array): Array of pupil width data.
            height (array): Array of pupil height data.
            x_saccade (array): Array of X-axis saccade data.
            y_saccade (array): Array of Y-axis saccade data.
        """

        # Convert width, height, and saccade data to numpy arrays
        width = np.array(width)
        height = np.array(height)
        self.app_instance.X_saccade_updated = np.array(x_saccade)
        self.app_instance.Y_saccade_updated = np.array(y_saccade)

        # Calculate the width-to-height ratio to identify potential blinking
        ratio = width / height

        # Detect blinking based on the ratio and pupil data using defined thresholds
        blinking_id_ratio = pupil_detection.detect_blinking_ids(ratio, 20)
        blinking_id_area = pupil_detection.detect_blinking_ids(pupil, 10)

        # Combine the detected blinking indices and remove duplicates
        combined_blinking_ids = list(set(blinking_id_ratio + blinking_id_area))

        # Ensure indices do not exceed the length of the saccade data
        combined_blinking_ids = [x for x in combined_blinking_ids if x < len(self.app_instance.X_saccade_updated[0])]
        combined_blinking_ids.sort()

        # Replace the detected blinking indices with NaNs in the updated saccade arrays
        self.app_instance.X_saccade_updated[0][combined_blinking_ids] = np.nan
        self.app_instance.Y_saccade_updated[0][combined_blinking_ids] = np.nan

        # Interpolate the pupil data to handle the blinking periods smoothly
        self.app_instance.interpolated_pupil = pupil_detection.interpolate(combined_blinking_ids, pupil)

        # Plot the interpolated pupil data with the detected saccade events
        self.app_instance.plot_handler.plot_result(
            self.app_instance.interpolated_pupil,
            self.app_instance.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.app_instance.X_saccade
        )

        # Store the final pupil area as the interpolated result
        self.app_instance.final_pupil_area = np.array(self.app_instance.interpolated_pupil)
        return combined_blinking_ids

    def pupil_dilation_comput(self, images, saturation, contrast, erased_pixels, reflect_ellipse, mnd, binary_threshold, clustering_method):
        """
        Computes pupil dilation and related metrics from a series of images using parallel processing.
        """
        # Start timing
        start_time = time.time()

        eye_corner_center = (
            self.app_instance.eye_corner_center[0],
            self.app_instance.eye_corner_center[1]
        ) if self.app_instance.eye_corner_center is not None else None


        ####################################################################
        pupil_ROI = self.app_instance.pupil_ROI
        sub_image = pupil_ROI.rect()
        top = int(sub_image.top())* self.app_instance.ratio
        bottom = int(sub_image.bottom())*self.app_instance.ratio
        left = int(sub_image.left())*self.app_instance.ratio
        right = int(sub_image.right())*self.app_instance.ratio
        frame = (top,bottom, left,right)
        ####################################################################
        # Prepare the data for multiprocessing
        frame_args = [
            (current_image, frame, saturation, contrast, erased_pixels, reflect_ellipse, eye_corner_center, mnd, binary_threshold,clustering_method)
            for current_image in images
        ]

        num_workers = max(1, cpu_count() // 2)

        print(f"Using {num_workers} workers out of {cpu_count()} cores.")

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_frame, frame_args), total=len(frame_args)))


        # Aggregate the results
        pupil_dilation, pupil_center, pupil_center_X, pupil_center_y, pupil_width, pupil_height, pupil_distance_from_corner = zip(
            *results
        )

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Time taken for pupil dilation computation: {elapsed_time:.2f} seconds")

        # Convert lists to numpy arrays for consistency and efficient computation
        pupil_dilation = np.array(pupil_dilation)
        pupil_center = np.array(pupil_center)
        pupil_center_X = np.array(pupil_center_X)
        pupil_center_y = np.array(pupil_center_y)
        pupil_width = np.array(pupil_width)
        pupil_height = np.array(pupil_height)
        pupil_distance_from_corner = np.array(pupil_distance_from_corner)
        #########################test for saccade###################

        # Compute saccades for X and Y coordinates
        X_saccade = self.Saccade(pupil_center_X)
        Y_saccade = self.Saccade(pupil_center_y)

        # Return all computed pupil metrics
        return (pupil_dilation, pupil_center_X, pupil_center_y, pupil_center,
                X_saccade, Y_saccade, pupil_distance_from_corner, pupil_width, pupil_height)


    def plot_saccade(self, pupil_center_i, saccade):
        """
        Plots the saccades based on pupil center changes.

        Parameters:
            pupil_center_i (array-like): The center coordinates of the pupil for each frame in the i-axis.
            saccade (array-like): Computed saccade values (differences with filtering).
        """
        # Frame indices for plotting
        frames = np.arange(len(pupil_center_i))

        # Plotting pupil center movement
        plt.figure(figsize=(10, 5))
        plt.plot(frames, pupil_center_i, label='Pupil Center (i-axis)', color='blue', marker='o', alpha=0.6)

        # Plotting saccades (filter out NaN values)
        plt.plot(frames, np.nan_to_num(saccade[0]), label='Saccade Movement', color='red', linestyle='--', alpha=0.7)

        # Adding labels, legend, and title
        plt.xlabel('Frame Index')
        plt.ylabel('Movement Value')
        plt.title('Saccade Movements Over Frames')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


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




