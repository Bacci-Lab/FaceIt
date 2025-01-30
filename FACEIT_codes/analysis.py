import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from FACEIT_codes import pupil_detection, functions
import math
import cv2
from multiprocessing import Pool
import time


# Example adjustment inside functions.py
def display_show_ROI(ROI, image):
    # Check if ROI is passed as a tuple (x, y, width, height)
    if isinstance(ROI, tuple):
        x, y, width, height = ROI
        sub_image = image[int(y):int(y + height), int(x):int(x + width)]
        return sub_image, ROI

    # Original logic if ROI is a QGraphicsEllipseItem
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
    current_image, pupil_roi_serializable, saturation, erased_pixels, reflect_ellipse, eye_corner_center = args

    # Reconstruct pupil ROI as a tuple or a simpler object (without PyQt)
    x, y, width, height = pupil_roi_serializable
    pupil_roi = (x, y, width, height)  # Replace this with the correct format expected by `show_ROI`

    # Process the image
    sub_region, _ = display_show_ROI(pupil_roi, current_image)  # Ensure `show_ROI` can handle this format

    if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
        sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)

    sub_region = functions.change_saturation(sub_region, saturation)
    sub_region_rgba = cv2.cvtColor(sub_region, cv2.COLOR_BGR2BGRA)

    _, center, width, height, _, current_area = pupil_detection.detect_pupil(
        sub_region_rgba, erased_pixels, reflect_ellipse
    )

    pupil_distance_from_corner = (
        math.sqrt((center[0] - eye_corner_center[0]) ** 2 + (center[1] - eye_corner_center[1]) ** 2)
        if eye_corner_center is not None else np.nan
    )

    return current_area, center, int(center[0]), int(center[1]), width, height, pupil_distance_from_corner



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
        start_time = time.time()
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
        # Duplicate the first computed value at the beginning of the list to avoid data length change
        if motion_energy_values:
            first_value = motion_energy_values[0]
            motion_energy_values.insert(0, first_value)
        end_time = time.time()
        print(f"Time taken for motion energy computation: {end_time - start_time:.2f} seconds")
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
        return combined_blinking_ids
    def extract_data_from_QGraphics(self,pupil_roi_data ):
        # Extract data from QGraphicsEllipseItem to make it serializable
        # Get the bounding rectangle (position and size)
        bounding_rect = pupil_roi_data.rect()
        x = bounding_rect.x()
        y = bounding_rect.y()
        width = bounding_rect.width()
        height = bounding_rect.height()

        # Convert to a simple, serializable tuple or dictionary
        pupil_roi_serializable = (x, y, width, height)
        return pupil_roi_serializable


    def pupil_dilation_comput(self, images, saturation, erased_pixels, reflect_ellipse):
        """
        Computes pupil dilation and related metrics from a series of images using parallel processing.
        """
        # Start timing
        start_time = time.time()

        # Extract only serializable data from PyQt objects before multiprocessing
        pupil_roi_data = self.app_instance.graphicsView_MainFig.pupil_ROI
        print("this is pupil_roi_data ", pupil_roi_data)
        eye_corner_center = (
            self.app_instance.eye_corner_center[0],
            self.app_instance.eye_corner_center[1]
        ) if self.app_instance.eye_corner_center is not None else None

        total_files = len(images)
        self.app_instance.progressBar.setMaximum(total_files)

        pupil_roi_serializable = self.extract_data_from_QGraphics(pupil_roi_data)

        # Prepare the data for multiprocessing
        frame_args = [
            (current_image, pupil_roi_serializable, saturation, erased_pixels, reflect_ellipse, eye_corner_center)
            for current_image in images
        ]

        # Use a multiprocessing pool to process images in parallel
        with Pool() as pool:
            results = pool.map(process_single_frame, frame_args)

        # Aggregate the results
        pupil_dilation, pupil_center, pupil_center_X, pupil_center_y, pupil_width, pupil_height, pupil_distance_from_corner = zip(
            *results
        )

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Time taken for pupil dilation computation: {elapsed_time:.2f} seconds")

        # Update progress bar to complete
        self.app_instance.progressBar.setValue(total_files)

        # Convert lists to numpy arrays for consistency and efficient computation
        pupil_dilation = np.array(pupil_dilation)
        pupil_center = np.array(pupil_center)
        pupil_center_X = np.array(pupil_center_X)
        pupil_center_y = np.array(pupil_center_y)
        pupil_width = np.array(pupil_width)
        pupil_height = np.array(pupil_height)
        pupil_distance_from_corner = np.array(pupil_distance_from_corner)
        #############test for saccade###################

        print("this pupil center", pupil_center)
        timestamps = np.arange(len(pupil_center))
        fixations_test, saccades_test = self.calculate_saccades_test(pupil_center,timestamps, 1)
        print("this is saccade test", saccades_test)

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
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line at y=0
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def calculate_saccades_test(self,positions, timestamps, velocity_threshold):
        """
        Detect saccades using the I-VT algorithm.

        Parameters:
        - positions: List of (x, y) pupil center positions [(x1, y1), (x2, y2), ...].
        - timestamps: List of timestamps corresponding to each position [t1, t2, ...].
        - velocity_threshold: Velocity threshold to classify fixations and saccades.

        Returns:
        - fixations: List of detected fixations, each as a dictionary with:
            {"x": centroid_x, "y": centroid_y, "start_time": t_start, "duration": duration}.
        - saccades: List of indices classified as saccades.
        """

        # Step 1: Calculate point-to-point velocities
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            velocity = np.sqrt(dx ** 2 + dy ** 2)
            velocities.append(velocity)

        # Step 2: Classify points as fixations or saccades
        classifications = [0]  # Start with the first point classified as fixation
        for v in velocities:
            if v < velocity_threshold:
                classifications.append(0)  # Fixation
            else:
                classifications.append(1)  # Saccade

        # Step 3: Collapse consecutive fixation points
        fixations = []
        saccades = []
        start_idx = None
        for i, cls in enumerate(classifications):
            if cls == 0:  # Fixation
                if start_idx is None:  # Start a new fixation group
                    start_idx = i
            else:  # Saccade
                if start_idx is not None:  # End the fixation group
                    fixation_points = positions[start_idx:i]
                    fixation_timestamps = timestamps[start_idx:i]
                    centroid_x = np.mean([p[0] for p in fixation_points])
                    centroid_y = np.mean([p[1] for p in fixation_points])
                    duration = fixation_timestamps[-1] - fixation_timestamps[0]
                    fixations.append({
                        "x": centroid_x,
                        "y": centroid_y,
                        "start_time": fixation_timestamps[0],
                        "duration": duration
                    })
                    start_idx = None  # Reset fixation group

        # Handle any remaining fixation group at the end
        if start_idx is not None:
            fixation_points = positions[start_idx:]
            fixation_timestamps = timestamps[start_idx:]
            centroid_x = np.mean([p[0] for p in fixation_points])
            centroid_y = np.mean([p[1] for p in fixation_points])
            duration = fixation_timestamps[-1] - fixation_timestamps[0]
            fixations.append({
                "x": centroid_x,
                "y": centroid_y,
                "start_time": fixation_timestamps[0],
                "duration": duration
            })

        # Identify saccade indices for completeness
        saccades = [i for i, cls in enumerate(classifications) if cls == 1]

        return fixations, saccades

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




