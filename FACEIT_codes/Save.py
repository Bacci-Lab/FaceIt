import numpy as np
from datetime import datetime
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries
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

        if hasattr(self.app_instance, 'pupil_center') and self.app_instance.pupil_center is not None:
            len_data = len(self.app_instance.pupil_center)
        else:
            len_data = len(self.app_instance.motion_energy)

        def initialize_data(attribute_name):
            setattr(self.app_instance, attribute_name, np.full((len_data,), np.nan))

        # Check and initialize pupil-related data
        if not self.app_instance.pupil_check():
            attributes = ['pupil_center', 'pupil_center_X', 'pupil_center_y', 'final_pupil_area','pupil_dilation',
                          'X_saccade_updated', 'Y_saccade_updated', 'pupil_distance_from_corner',
                          'width', 'height']
            for attr in attributes:
                initialize_data(attr)

        # Check and initialize face-related data
        if not self.app_instance.face_check():
            self.app_instance.motion_energy = np.full((len_data,), np.nan)
            self.app_instance.facemotion_without_grooming = np.full((len_data,), np.nan)

        # Save data
        self.save_data(
            pupil_center=self.app_instance.pupil_center,
            pupil_center_X=self.app_instance.pupil_center_X,
            pupil_center_y=self.app_instance.pupil_center_y,
            pupil_dilation_blinking_corrected=self.app_instance.final_pupil_area,
            pupil_dilation= self.app_instance.pupil_dilation,
            X_saccade=self.app_instance.X_saccade_updated,
            Y_saccade=self.app_instance.Y_saccade_updated,
            pupil_distance_from_corner=self.app_instance.pupil_distance_from_corner,
            width=self.app_instance.width,
            height=self.app_instance.height,
            motion_energy=self.app_instance.motion_energy,
            motion_energy_without_grooming = self.app_instance.facemotion_without_grooming
        )

    def save_data(self, **data_dict):
        """
        Saves data to a .npz and nwb file

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
        try:
            nwb_save_path = os.path.join(self.app_instance.save_path, "faceit.nwb")
            self.save_nwb(nwb_save_path)
        except Exception as e:
            logging.error(f"Failed to save .nwb data: {e}")

    def save_nwb(self, output_path):
        """
        Saves data to an .nwb file at the specified path.

        Parameters:
        output_path (str): The path where the .nwb file will be saved.
        """
        if self.app_instance.nwb_check():
            nwbfile = NWBFile(
                session_description='faceit',
                identifier='pupil_dilation_data',
                session_start_time=datetime.now(),
                file_create_date=datetime.now(),
            )

            if hasattr(self.app_instance, 'pupil_center') and self.app_instance.pupil_center is not None:
                len_data = len(self.app_instance.pupil_center)
            else:
                len_data = len(self.app_instance.motion_energy)

            def initialize_data(attribute_name):
                setattr(self.app_instance, attribute_name, np.full((len_data,), np.nan))

            # Check and initialize pupil-related data
            if not self.app_instance.pupil_check():
                attributes = ['pupil_center', 'pupil_center_X', 'pupil_center_y', 'final_pupil_area','pupil_dilation',
                              'X_saccade_updated', 'Y_saccade_updated', 'pupil_distance_from_corner',
                              'width', 'height']
                for attr in attributes:
                    initialize_data(attr)

            # Check and initialize face-related data
            if not self.app_instance.face_check():
                self.app_instance.motion_energy = np.full((len_data,), np.nan)
                self.app_instance.facemotion_without_grooming = np.full((len_data,), np.nan)

            time_stamps = np.arange(0, len_data, 1)  #time stamps in frame


            pupil_center_series = TimeSeries(
                name='pupil_center',
                data=self.app_instance.pupil_center,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            pupil_center_X_series = TimeSeries(
                name='pupil_center_X',
                data=self.app_instance.pupil_center_X,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            pupil_center_y_series = TimeSeries(
                name='pupil_center_y',
                data=self.app_instance.pupil_center_y,
                unit='arbitrary units',
                timestamps=time_stamps
            )


            pupil_dilation_blinking_corrected_series = TimeSeries(
                name='pupil_dilation_blinking_corrected',
                data=self.app_instance.final_pupil_area,
                unit='arbitrary units',
                timestamps=time_stamps
            )

            pupil_dilation_series = TimeSeries(
                name='pupil_dilation',
                data=self.app_instance.pupil_dilation,
                unit='arbitrary units',
                timestamps=time_stamps
            )

            X_saccade_series = TimeSeries(
                name='X_saccade',
                data=self.app_instance.X_saccade_updated,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            Y_saccade_series = TimeSeries(
                name='Y_saccade',
                data=self.app_instance.Y_saccade_updated,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            pupil_distance_from_corner_series = TimeSeries(
                name='pupil_distance_from_corner',
                data=self.app_instance.pupil_distance_from_corner,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            width_series = TimeSeries(
                name='width',
                data=self.app_instance.width,
                unit='arbitrary units',
                timestamps=time_stamps
            )

            height_series = TimeSeries(
                name='height',
                data=self.app_instance.height,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            motion_energy_series = TimeSeries(
                name='motion_energy',
                data=self.app_instance.motion_energy,
                unit='arbitrary units',
                timestamps=time_stamps
            )
            motion_energy_without_grooming_series = TimeSeries(
                name='motion_energy_without_grooming',
                data=self.app_instance.facemotion_without_grooming,
                unit='arbitrary units',
                timestamps=time_stamps
            )


            processing_module = ProcessingModule(
                name='eye facial movement',
                description='Contains pupil size, dilation, position and saccade data and whisker pad motion energy'
            )

            nwbfile.add_processing_module(processing_module)

            # Add the TimeSeries data to the processing module
            processing_module.add_data_interface(pupil_dilation_series)
            processing_module.add_data_interface(pupil_dilation_blinking_corrected_series)
            processing_module.add_data_interface(pupil_center_series)
            processing_module.add_data_interface(pupil_center_y_series)
            processing_module.add_data_interface(pupil_center_X_series)
            processing_module.add_data_interface(X_saccade_series)
            processing_module.add_data_interface(Y_saccade_series)
            processing_module.add_data_interface(width_series)
            processing_module.add_data_interface(height_series)
            processing_module.add_data_interface(pupil_distance_from_corner_series)
            processing_module.add_data_interface(motion_energy_series)
            processing_module.add_data_interface(motion_energy_without_grooming_series)

            with NWBHDF5IO(output_path, 'w') as io:
                io.write(nwbfile)

            print(f"Data successfully saved to {output_path}")
        else:
            print("NWB check failed; data not saved.")