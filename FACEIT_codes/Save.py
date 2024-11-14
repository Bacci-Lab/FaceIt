import numpy as np
import os
import logging

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SaveHandler:
    def __init__(self, app_instance):
        """
        Initialize the SaveHandler class with a reference to the app instance.

        Parameters:
        app_instance (object): The main application instance containing relevant data and methods.
        """
        self.app_instance = app_instance

    def init_save_data(self):
        """
        Initializes data for saving. Checks if pupil and face data exist;
        if not, it fills the arrays with NaNs.
        """
        len_data = 100

        # Helper function to initialize data arrays with NaNs
        def initialize_data(attribute_name):
            setattr(self.app_instance, attribute_name, np.full((len_data,), np.nan))

        # Check and initialize pupil-related data
        if not self.app_instance.pupil_check():
            attributes = ['pupil_center', 'pupil_center_X', 'pupil_center_y', 'final_pupil_area',
                          'X_saccade_updated', 'Y_saccade_updated', 'pupil_distance_from_corner',
                          'width', 'height']
            for attr in attributes:
                initialize_data(attr)

        # Check and initialize face-related data
        if not self.app_instance.face_check():
            self.app_instance.motion_energy = np.full((len_data,), np.nan)

        # Save data
        self.save_data(
            pupil_center=self.app_instance.pupil_center,
            pupil_center_X=self.app_instance.pupil_center_X,
            pupil_center_y=self.app_instance.pupil_center_y,
            pupil_dilation=self.app_instance.final_pupil_area,
            X_saccade=self.app_instance.X_saccade_updated,
            Y_saccade=self.app_instance.Y_saccade_updated,
            pupil_distance_from_corner=self.app_instance.pupil_distance_from_corner,
            width=self.app_instance.width,
            height=self.app_instance.height,
            motion_energy=self.app_instance.motion_energy
        )

    def save_data(self, **data_dict):
        """
        Saves data to a .npz file.

        Parameters:
        data_dict (dict): Key-value pairs where keys are data attribute names and values are data arrays.
        """
        save_directory = os.path.join(self.app_instance.save_path, "faceit.npz")

        try:
            # Save the data dictionary as a compressed .npz file
            np.savez(save_directory, **data_dict)
            logging.info(f"Data successfully saved to {save_directory}")
        except Exception as e:
            logging.error(f"Failed to save data: {e}")
