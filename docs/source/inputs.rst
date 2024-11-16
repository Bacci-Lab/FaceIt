=======
Inputs
=======

The FaceIt pipeline accepts and processes various types of input data, primarily image files and video files.

Input Types Overview
--------------------

1. **Image Folders (.npy Files)**

   - The pipeline supports directories containing ``.npy`` image files. Each ``.npy`` file should represent a single image or frame to be processed.


2. **Video Files (.avi)**

   - The pipeline also accepts ``.avi`` video files as input.

Important Note
--------------

- **Directory Structure**: The images should be in a single directory. Users can select the directory using the graphical interface provided by the application.

- **Important**: ``.npy`` image files should be named in a way that they have the correct temporal sequence (e.g., `frame_001.npy`, `frame_002.npy`, etc.) to ensure proper chronological order during processing.

