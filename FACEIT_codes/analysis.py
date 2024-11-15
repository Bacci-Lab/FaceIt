import numpy as np
from PyQt5 import QtWidgets
from FACEIT_codes import pupil_detection
from FACEIT_codes import functions
import math
import cv2
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
        """Processes and plots pupil data."""
        if not self.app_instance.Pupil_ROI_exist:
            # Warn the user if no Pupil ROI is chosen
            self.app_instance.warning("NO Pupil ROI is chosen!")
            return

        # Ensure images are loaded
        self.load_images_if_needed()

        # Compute pupil data
        self.app_instance.pupil_dilation, self.app_instance.pupil_center_X, self.app_instance.pupil_center_y, \
            self.app_instance.pupil_center, self.app_instance.X_saccade, self.app_instance.Y_saccade, \
            self.app_instance.pupil_distance_from_corner, self.app_instance.width, self.app_instance.height = \
            self.app_instance.start_pupil_dilation_computation(self.app_instance.images)

        # Plot the result
        self.app_instance.plot_handler.plot_result(
            self.app_instance.pupil_dilation,
            self.app_instance.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.app_instance.X_saccade
        )

    def process_face_data(self):
        """Processes and plots face data."""
        if not self.app_instance.Face_ROI_exist:
            # Warn the user if no Face ROI is chosen
            self.app_instance.warning("NO Face ROI is chosen!")
            return

        # Ensure images are loaded
        self.load_images_if_needed()

        # Compute motion energy for face
        self.app_instance.motion_energy = self.motion_Energy_comput(self.app_instance.images)

        # Plot the result
        self.app_instance.plot_handler.plot_result(
            self.app_instance.motion_energy,
            self.app_instance.graphicsView_whisker,
            "motion"
        )

    def motion_Energy_comput(self, images):
        """
        Computes the motion energy from a series of image frames.

        Parameters:
            images (list): A list of image arrays.

        Returns:
            list: A list of computed motion energy values.
        """
        frame = self.app_instance.Face_frame
        motion_energy_values = []

        # Total number of image frames to process
        total_files = len(images)
        self.app_instance.progressBar.setMaximum(total_files)

        # Variable to store the previous frame's region of interest (ROI)
        previous_ROI = None

        # Iterate through each image and compute motion energy
        for i, current_array in enumerate(images):
            # Update progress bar and process UI events
            self.app_instance.progressBar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

            # Extract the region of interest (ROI) from the current frame
            current_ROI = current_array[frame[0]:frame[1], frame[2]:frame[3]].flatten()

            # Compute motion energy if the previous ROI exists
            if previous_ROI is not None:
                motion_energy_value = np.mean((current_ROI - previous_ROI) ** 2)
                motion_energy_values.append(motion_energy_value)

            # Update the previous ROI for the next iteration
            previous_ROI = current_ROI

        # Ensure the progress bar is set to complete
        self.app_instance.progressBar.setValue(total_files)

        return motion_energy_values
    def load_images_if_needed(self):
        """Loads images if they are not already loaded."""
        if not self.app_instance.Image_loaded:
            if self.app_instance.NPY:
                # Load images from a directory of .npy files
                self.app_instance.images = self.app_instance.load_handler.load_images_from_directory(self.app_instance.folder_path)
            elif self.app_instance.video:
                # Load images from a video file
                self.app_instance.images = self.app_instance.load_handler.load_frames_from_video(self.app_instance.folder_path)

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
    def pupil_dilation_comput(self, images, saturation, blank_ellipse, reflect_ellipse):
        """
        Computes pupil dilation and related metrics from a series of images.

        Parameters:
            images (list): List of image arrays.
            saturation (int): Saturation level for image processing.
            blank_ellipse (tuple): Parameters for blank ellipse adjustment.
            reflect_ellipse (tuple): Parameters for reflection ellipse adjustment.

        Returns:
            tuple: Computed pupil dilation, center coordinates (X and Y),
                   pupil center, saccade data (X and Y), distance from the corner,
                   pupil width, and height.
        """

        # Retrieve the pupil region of interest (ROI)
        pupil = self.app_instance.graphicsView_MainFig.pupil_ROI

        # Initialize lists to store computed data
        total_files = len(images)
        pupil_dilation = []
        pupil_center_X = []
        pupil_center_y = []
        pupil_center = []
        pupil_width = []
        pupil_height = []

        # Set progress bar maximum to the number of images
        self.app_instance.progressBar.setMaximum(total_files)

        # Process each image in the list
        for i, current_image in enumerate(images):
            # Update progress bar and process UI events to keep the interface responsive
            self.app_instance.progressBar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

            # Extract the pupil sub-region from the current image
            sub_region, _ = functions.show_ROI(pupil, current_image)

            # Convert grayscale images to RGB format if necessary
            if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
                sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)

            # Apply saturation adjustment
            sub_region = functions.change_saturation(sub_region, saturation)

            # Convert to RGBA format for further processing
            sub_region_rgba = cv2.cvtColor(sub_region, cv2.COLOR_BGR2BGRA)

            # Detect pupil characteristics from the processed sub-region
            _, center, width, height, _, current_area = functions.detect_pupil(
                sub_region_rgba, blank_ellipse, reflect_ellipse
            )

            # Store computed metrics for the current frame
            pupil_width.append(width)
            pupil_height.append(height)
            pupil_dilation.append(current_area)
            pupil_center.append(center)
            pupil_center_X.append(int(center[0]))
            pupil_center_y.append(int(center[1]))

        # Set progress bar to complete after processing all images
        self.app_instance.progressBar.setValue(total_files)

        # Convert lists to numpy arrays for consistency and efficient computation
        pupil_dilation = np.array(pupil_dilation)
        pupil_center_X = np.array(pupil_center_X)
        pupil_center_y = np.array(pupil_center_y)
        pupil_center = np.array(pupil_center)
        pupil_width = np.array(pupil_width)
        pupil_height = np.array(pupil_height)

        # Compute saccades for X and Y coordinates
        X_saccade = self.Saccade(pupil_center_X)
        Y_saccade = self.Saccade(pupil_center_y)

        # Calculate the distance from the pupil center to the eye corner, if defined
        if self.app_instance.eye_corner_center is not None:
            pupil_distance_from_corner = np.array([
                math.sqrt((x - self.app_instance.eye_corner_center[0]) ** 2 + (y - self.app_instance.eye_corner_center[1]) ** 2)
                for x, y in pupil_center
            ])
        else:
            # If eye corner center is not defined, fill with NaNs
            pupil_distance_from_corner = np.full((len(X_saccade),), np.nan)

        # Return all computed pupil metrics
        return (pupil_dilation, pupil_center_X, pupil_center_y, pupil_center,
                X_saccade, Y_saccade, pupil_distance_from_corner, pupil_width, pupil_height)

    def Saccade(self, pupil_center_i):
        """
        Computes the saccade movements based on changes in the pupil center coordinates.

        Parameters:
            pupil_center_i (array-like): The center coordinates of the pupil for each frame.

        Returns:
            np.ndarray: A 2D array with saccade values, where small changes (<2) are replaced with NaN.
        """
        # Compute saccade differences using numpy's vectorized operations
        saccade = np.diff(pupil_center_i).astype(float)

        # Set small movements (absolute value < 2) to NaN for better filtering of significant movements
        saccade[abs(saccade) < 2] = np.nan

        # Reshape the saccade array to be 2D (1 row) for consistency
        saccade = saccade.reshape(1, -1)

        return saccade


