import numpy as np
from datetime import datetime
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries
import os
import logging
import matplotlib.pyplot as plt
from pynwb.image import ImageSeries
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

        if self.app_instance.save_video_chack():
            pass

        self.save_directory = self._make_dir(self.app_instance.save_path)

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
        if not hasattr(self, 'facemotion_without_grooming') or self.facemotion_without_grooming is None:
            self.app_instance.facemotion_without_grooming = self.app_instance.motion_energy
            self.app_instance.grooming_ids = np.full((len_data,), np.nan)
            self.app_instance.grooming_thr = np.full(1, np.nan)
        if not hasattr(self.app_instance, 'blinking_ids'):
            self.app_instance.blinking_ids = np.full((len_data,), np.nan)
        else:
            pass


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
            motion_energy_without_grooming = self.app_instance.facemotion_without_grooming,
            grooming_ids = self.app_instance.grooming_ids,
            grooming_threshold = self.app_instance.grooming_thr,
            blinking_ids = self.app_instance.blinking_ids
        )

    def save_data(self, **data_dict):
        """
        Saves data to a .npz file and includes a video file as binary data if 'save_video' is checked.
        """
        save_directory = os.path.join(self.save_directory, "faceit.npz")

        # Check if "save_video" is checked
        if self.app_instance.save_video_chack():
            print("Saving video data...")
            video_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\FaceCamera.wmv"

            if os.path.exists(video_path):
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()  # Read video as binary
                    data_dict["video_file"] = np.array([video_bytes], dtype=object)
            else:
                print(f"Video file not found at {video_path}")

        try:
            np.savez_compressed(save_directory, **data_dict)
            logging.info(f"Data successfully saved to {save_directory}")
        except Exception as e:
            logging.error(f"Failed to save data: {e}")
        try:
            nwb_save_path = os.path.join(self.save_directory, "faceit.nwb")
            self.save_nwb(nwb_save_path)
        except Exception as e:
            logging.error(f"Failed to save .nwb data: {e}")
        try:
            self.save_fig()
        except Exception as e:
            logging.error(f"Failed to save image: {e}")


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
            if not hasattr(self, 'facemotion_without_grooming') or self.facemotion_without_grooming is None:
                self.app_instance.facemotion_without_grooming = self.app_instance.motion_energy

            time_stamps = np.arange(0, len_data, 1)

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

            grooming_id_series = TimeSeries(
                name='grooming ids',
                data=self.app_instance.grooming_ids,
                unit='frame',
                timestamps=time_stamps
            )

            grooming_threshold = TimeSeries(
                name='grooming threshold',
                data=self.app_instance.grooming_thr,
                unit='frame',
                timestamps=time_stamps
            )

            blinking_ids = TimeSeries(
                name='blinking ids',
                data=self.app_instance.blinking_ids,
                unit='frame',
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
            processing_module.add_data_interface(blinking_ids)
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
            processing_module.add_data_interface(grooming_id_series)
            processing_module.add_data_interface(grooming_threshold)


            with NWBHDF5IO(output_path, 'w') as io:
                io.write(nwbfile)

            print(f"Data successfully saved to {output_path}")
        else:
            print("NWB check failed; data not saved.")

    def save_fig(self):
        """
        Saves figures for pupil dilation and motion energy data.

        """
        # Check and save pupil dilation data
        if hasattr(self.app_instance, 'pupil_dilation') and self.app_instance.pupil_dilation is not None:
            self._save_single_fig(
                data=self.app_instance.pupil_dilation,
                label='pupil_dilation',
                color='palegreen',
                filename="pupil_area.png",
                saccade_data=self.app_instance.X_saccade_updated
            )

        # Check and save motion energy data
        if hasattr(self.app_instance, 'motion_energy') and self.app_instance.motion_energy is not None:
            self._save_single_fig(
                data=self.app_instance.motion_energy,
                label='motion_energy',
                color='salmon',
                filename="motion_energy.png"
            )

    def _save_single_fig(self, data, label, color, filename, saccade_data = None):
        """
        Helper function to plot data and save a figure.

        Parameters:
        data (np.ndarray): The main data to plot.
        label (str): The label for the plot.
        color (str): The color of the plot line.
        filename (str): The name of the file to save the plot as.
        saccade_data (np.ndarray, optional): Saccade data to overlay on the plot.
        """
        fig, ax = plt.subplots()
        save_path = os.path.join(self.save_directory, filename)
        self._plot_data(ax, data, label, color)

        # Plot saccade data if provided
        if saccade_data is not None:
            self._plot_saccade(ax, saccade_data, data)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def _plot_data(self, ax: plt.Axes, data: np.ndarray, label: str, color: str):
        """
        Plots the main data on the provided axes.

        Parameters:
        ax (plt.Axes): The axes to plot on.
        data (np.ndarray): The data to plot.
        label (str): The label for the plot line.
        color (str): The color of the plot line.
        """
        x_values = np.arange(len(data))
        ax.plot(x_values, data, color=color, label=label, linestyle='--')

    def _plot_saccade(self, ax: plt.Axes, saccade: np.ndarray, data: np.ndarray):
        """
        Plots the saccade data as a colormap overlay if provided.

        Parameters:
        ax (plt.Axes): The axes to plot on.
        saccade (np.ndarray): The saccade data to overlay.
        data (np.ndarray): The main data array for reference to calculate plot boundaries.
        """
        if saccade is not None:
            if saccade.shape[1] == len(data):
                saccade = saccade[:, 1:]  # Trim the first column to match the length
            data_max = np.max(data)
            range_val = np.max(data) - np.min(data)
            y_min = data_max + range_val / 10
            y_max = data_max + range_val / 5
            x_values = np.arange(len(data))
            ax.pcolormesh(x_values, [y_min, y_max], saccade, cmap='RdYlGn', shading='flat')

    def _make_dir(self, base_path):
        save_directory = os.path.join(base_path, "FaceIt")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        return save_directory
