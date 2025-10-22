from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

nwb_path = Path(r"C:\Users\faezeh.rabbani\FaceIt_saves\FaceIt/faceit.nwb")

with NWBHDF5IO(str(nwb_path), "r") as io:
    nwb = io.read()
    mod = nwb.processing["eye_facial_movement"]

    # List series names:
    print(list(mod.data_interfaces.keys()))

    # Read common series
    pupil_dilation = np.asarray(mod.data_interfaces["pupil_dilation"].data)
    pupil_dilation_blinking_corrected = np.asarray(
        mod.data_interfaces["pupil_dilation_blinking_corrected"].data
    )
    blinking_ids = np.asarray(mod.data_interfaces["blinking_ids"].data)
    pupil_center = np.asarray(mod.data_interfaces["pupil_center"].data)
    pupil_center_X = np.asarray(mod.data_interfaces["pupil_center_X"].data)
    pupil_center_y = np.asarray(mod.data_interfaces["pupil_center_y"].data)
    X_saccade = np.asarray(mod.data_interfaces["X_saccade"].data)
    Y_saccade = np.asarray(mod.data_interfaces["Y_saccade"].data)
    pupil_width = np.asarray(mod.data_interfaces["width"].data)
    pupil_height = np.asarray(mod.data_interfaces["height"].data)
    motion_energy = np.asarray(mod.data_interfaces["motion_energy"].data)
    motion_energy_without_grooming = np.asarray(
        mod.data_interfaces["motion_energy_without_grooming"].data
    )
    grooming_ids = np.asarray(mod.data_interfaces["grooming_ids"].data)
    grooming_threshold = np.asarray(mod.data_interfaces["grooming_threshold"].data)
    pupil_distance_from_corner = np.asarray(
        mod.data_interfaces["pupil_distance_from_corner"].data
    )
    pupil_angle = np.asarray(mod.data_interfaces["pupil_angle"].data)
    print("pupil_angle", pupil_angle)