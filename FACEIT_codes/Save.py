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

    # def save_nwb():
    #
    #     # Step 1: Create an NWBFile object with the required metadata
    #     nwbfile = NWBFile(
    #         session_description='Pupil dilation experiment',  # Description of the experiment
    #         identifier='pupil_dilation_data',  # Unique ID for this session
    #         session_start_time=datetime.now(),  # Time when the session started
    #         file_create_date=datetime.now(),  # Creation time of the file
    #     )
    #
    #     # Step 2: Add pupil dilation data (as a time series)
    #     # Replace this with your actual pupil dilation data
    #     pupil_dilation = np.random.rand(1000)  # Dummy data for pupil dilation
    #     saccade_data = np.random.rand(1000)  # Dummy data for saccade
    #
    #     # Time in seconds
    #     time_stamps = np.arange(0, 1000, 1) * 0.001  # Example time stamps in seconds
    #
    #     # Create TimeSeries objects for pupil dilation and saccade
    #     from pynwb.base import TimeSeries
    #
    #     pupil_dilation_series = TimeSeries(
    #         name='Pupil Dilation',
    #         data=pupil_dilation,
    #         unit='arbitrary units',  # Replace with appropriate unit (e.g., 'mm' for pupil diameter)
    #         timestamps=time_stamps
    #     )
    #
    #     saccade_series = TimeSeries(
    #         name='Saccade',
    #         data=saccade_data,
    #         unit='arbitrary units',  # Replace with appropriate unit
    #         timestamps=time_stamps
    #     )
    #
    #     # Add TimeSeries data to the NWBFile
    #     nwbfile.add_acquisition(pupil_dilation_series)
    #     nwbfile.add_acquisition(saccade_series)
    #
    #     # Step 3: Write the NWB file to disk
    #     output_filename = 'pupil_dilation_data.nwb'
    #     with NWBHDF5IO(output_filename, 'w') as io:
    #         io.write(nwbfile)
    #
    #     print(f"Data successfully saved to {output_filename}")
