Outputs
========

The FaceIt pipeline generates multiple outputs stored in `.npz` and `.nwb` files, plus visualization images. This page describes what each file contains and how to read them safely.

Outputs Overview
----------------

`.npz` file (compressed data archive)
-------------------------------------

Saved as ``faceit.npz``. It may include the following arrays (keys):

- ``pupil_center``: Pupil center (per-frame; typically 1D or structured).
- ``pupil_center_X``, ``pupil_center_y``: X and Y coordinates of the pupil center.
- ``pupil_dilation_blinking_corrected``: Pupil area/dilation after blink correction.
- ``pupil_dilation``: Raw pupil area/dilation.
- ``X_saccade``, ``Y_saccade``: Saccade matrices (often ``2×T`` or heatmap-like overlays).
- ``pupil_distance_from_corner``: Distance of pupil center to eye corner (per frame).
- ``width``, ``height``: Fitted ellipse dimensions (per frame).
- ``motion_energy``: Whisker pad (face) motion energy (per frame).
- ``motion_energy_without_grooming``: Motion energy with grooming filtered out.
- ``grooming_ids``: Frame indices flagged as grooming.
- ``grooming_threshold``: Grooming threshold value(s).
- ``blinking_ids``: Frame indices flagged as blinks.
- ``angle``: Fitted pupil ellipse angle (per frame).
- ``Face_frame``, ``Pupil_frame``: Frame reference values (scalar/short arrays).

.. note::
   Keys are snake_case with underscores (e.g., ``grooming_ids``, not ``"grooming ids"``).



`.nwb` file (Neurodata Without Borders)
--------------------------------------

Saved as ``faceit.nwb`` if the "Save NWB" option is checked. TimeSeries are stored under a
``ProcessingModule`` named ``eye_facial_movement``. The series mirror the ``.npz`` content:

**TimeSeries (typical keys):**
- ``pupil_center``, ``pupil_center_X``, ``pupil_center_y``
- ``pupil_dilation_blinking_corrected``, ``pupil_dilation``
- ``X_saccade``, ``Y_saccade``
- ``pupil_distance_from_corner``
- ``width``, ``height``
- ``motion_energy``, ``motion_energy_without_grooming``
- ``grooming_ids``, ``grooming_threshold``
- ``blinking_ids``
- ``pupil_angle``
- ``Face_frame``, ``Pupil_frame``

.. note::
   Each TimeSeries uses timestamps sized to its own length.


Visualization images (``.png``)
-------------------------------

The pipeline saves quick-look plots in the save directory:

- ``blinking_corrected.png`` — blink-corrected pupil area
- ``pupil_area.png`` — raw pupil area/dilation
- ``motion_energy.png`` — motion energy (with grooming threshold line if available)
- ``facemotion_without_grooming.png`` — motion energy with grooming removed


Access to data
--------------

**Example: read ``.nwb`` (and list available series)**

.. code:: python

    from pathlib import Path
    import numpy as np
    from pynwb import NWBHDF5IO

    nwb_path = Path("path/to/faceit.nwb")

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


**Example: read ``.npz`` (and handle optional embedded video)**

.. code:: python

    from pathlib import Path
    import numpy as np

    npz_path = Path("path/to/faceit.npz")

    # allow_pickle=True is needed if the file contains object arrays (e.g., 'video_file')
    with np.load(npz_path, allow_pickle=True) as z:
        print("Keys:", list(z.files))

        pupil_center = z["pupil_center"]
        motion_energy = z["motion_energy"]
        pupil_dilation = z["pupil_dilation"]
        pupil_dilation_blinking_corrected = z["pupil_dilation_blinking_corrected"]
        pupil_center_X = z["pupil_center_X"]
        pupil_center_y = z["pupil_center_y"]
        X_saccade = z["X_saccade"]
        Y_saccade = z["Y_saccade"]
        pupil_width = z["width"]
        pupil_height = z["height"]
        motion_energy_without_grooming = z["motion_energy_without_grooming"]
        grooming_ids = z["grooming_ids"]
        grooming_threshold = z["grooming_threshold"]
        blinking_ids = z["blinking_ids"]
        pupil_distance_from_corner = z["pupil_distance_from_corner"]
        angle = z["angle"]
        Face_frame = z["Face_frame"]
        Pupil_frame = z["Pupil_frame"]

        # Optional: extract embedded video if present
        if "video_file" in z.files:
            video_bytes = z["video_file"][0]
            out_vid = npz_path.with_suffix("").parent / (npz_path.stem + "_embedded_video.wmv")
            with open(out_vid, "wb") as f:
                f.write(video_bytes)
            print(f"Extracted video to: {out_vid}")


Tips & Requirements
-------------------

- Install:
  - **NumPy** to read ``.npz``.
  - **PyNWB** (and dependencies like **h5py**) to read ``.nwb``.

- Keys and names are **case- and underscore-sensitive**:
  - Use ``pupil_center_y`` (lowercase ``y``), not ``pupil_center_Y``.
  - Use ``grooming_ids`` / ``blinking_ids``, not names with spaces.
