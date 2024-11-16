========================
Output of the FaceIt Pipeline
========================

The FaceIt pipeline generates multiple outputs that are stored in different formats, including `.npz` and `.nwb` files, as well as visualization images.

Outputs Overview
----------------

1. **.npz File (Compressed Data Archive)**
   - The `.npz` file is saved as `faceit.npz` and contains the following key data arrays:
     - **pupil_center**: Coordinates of the pupil center across frames.
     - **pupil_center_X** and **pupil_center_y**: X and Y coordinates of the pupil center.
     - **pupil_dilation_blinking_corrected**: Corrected pupil dilation values, accounting for blinking artifacts.
     - **blinking ids**: Indices indicating frames where blinking activities were detected.
     - **pupil_dilation**: Raw pupil dilation data.
     - **X_saccade** and **Y_saccade**: Saccadic movements in the X and Y directions.
     - **pupil_distance_from_corner**: Distance of the pupil center from eye corner.
     - **width** and **height**: Dimensions of the detected pupil.
     - **motion_energy**: Motion energy values computed across frames.
     - **motion_energy_without_grooming**: Motion energy data with grooming movements filtered out.
     - **grooming_ids**: Indices indicating frames where grooming activities were detected.
     - **grooming_threshold**: Threshold value determined by user for identifying grooming activities.



2. **.nwb File (Neurodata Without Borders)**
   - The `.nwb` file, saved as `faceit.nwb`, contains structured time-series data for advanced data handling and analysis:
     - **TimeSeries Data**:
       - **`pupil_center`**, **`pupil_center_X`**, **`pupil_center_y`**
       - **`pupil_dilation_blinking_corrected`**, **`pupil_dilation`**
       - **`X_saccade`**, **`Y_saccade`**
       - **`pupil_distance_from_corner`**, **blinking ids**
       - **`width`**, **`height`**
       - **`motion_energy`**, **`motion_energy_without_grooming`**
       - **`grooming ids`**, **`grooming_threshold`**

   - These data series are stored within a `ProcessingModule` labeled **"eye facial movement"**, which includes attributes relevant to pupil and facial motion analysis. The content of each time series is the same as **".npz"** file.




3. **Visualization Images**
   - The pipeline generates visual plots that are saved as `.png` images in the designated directory:
     - **`pupil_area.png`**: A plot showing the pupil dilation data over the frame sequence.
     - **`motion_energy.png`**: A plot representing the motion energy over the frame sequence.
   - These images provide quick visual references for understanding data trends.

Access to data
--------------
- Example of reading the `.nwb` file:

.. code:: python

     from pynwb import NWBHDF5IO

     with NWBHDF5IO('path/to/faceit.nwb', 'r') as io:
         nwbfile = io.read()
         processing = nwbfile.processing['eye facial movement']
         pupil_dialtion = processing.data_interfaces['pupil_dilation'].data[:]
         pupil_dilation_blinking_corrected = processing.data_interfaces['pupil_dilation_blinking_corrected'].data[:]
         blinking_ids = processing.data_interfaces['blinking ids'].data[:]
         pupil_center = processing.data_interfaces['pupil_center'].data[:]
         pupil_center_X = processing.data_interfaces['pupil_center_X'].data[:]
         pupil_center_Y = processing.data_interfaces['pupil_center_Y'].data[:]
         X_saccade = processing.data_interfaces['X_saccade'].data[:]
         Y_saccade = processing.data_interfaces['Y_saccade'].data[:]
         pupil_width = processing.data_interfaces['width'].data[:]
         pupil_height = processing.data_interfaces['height'].data[:]
         motion_energy = processing.data_interfaces['motion_energy'].data[:]
         motion_energy_without_grooming = processing.data_interfaces['motion_energy_without_grooming'].data[:]
         grooming_ids = processing.data_interfaces['grooming ids'].data[:]
         grooming_threshold = processing.data_interfaces['grooming_threshold'].data[:]
         pupil_distance_from_corner = processing.data_interfaces['pupil_distance_from_corner'].data[:]


- Example of reading data in .npz file:


.. code:: python

     import numpy as np
     data = np.load('path/to/faceit.npz')
     pupil_center = data['pupil_center']
     motion_energy = data['motion_energy']
     pupil_dialtion = data['pupil_dilation']
     pupil_dilation_blinking_corrected = data['pupil_dilation_blinking_corrected']
     pupil_center_X = data['pupil_center_X']
     pupil_center_Y = data['pupil_center_Y']
     X_saccade = data['X_saccade']
     Y_saccade = data['Y_saccade']
     pupil_width = data['width']
     pupil_height = data['height']
     motion_energy = data['motion_energy']
     motion_energy_without_grooming = data['motion_energy_without_grooming']
     grooming_ids = data['grooming ids']
     grooming_threshold = data['grooming_threshold']
     pupil_distance_from_corner = data['pupil_distance_from_corner']


Details and Requirements
------------------------
To use the output generated by the FaceIt pipeline, you can easily access and load the data using Python. This guide explains how to read both the ``'.npz'`` and ``'.nwb'`` file formats, which are produced by the pipeline.

Ensure that the following Python packages are installed:

NumPy: To read .npz files.

PyNWB: To read .nwb files.

