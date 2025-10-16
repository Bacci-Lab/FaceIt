How to Use FaceIt GUI
=====================

Open data to process
^^^^^^^^^^^^^^^^^^^^

To analyze mouse face motion data, navigate to the File tab in the menu bar and select the option to open an image series **Folder** (Ctrl + N) or a video  **file** (Ctrl + V). Once selected, the images or video frames will be displayed in the GUI for analysis.


Analyse Pupil
^^^^^^^^^^^^^^

To begin pupil tracking, use the **Pupil ROI** button in the **ROI tools** section to define the eyeball region. You can adjust the position of the Pupil ROI by dragging it or resize it by clicking and dragging the blue square at its corner.

.. image:: _static/ROI_tools.png
   :alt: Image ROI_tools
   :width: 400px
   :align: center

.. raw:: html

   <div style="margin-bottom: 20px;"></div>

Pupil_roi_chosen
After selecting and adjusting the Eyeball ROI, you can use the **Eraser** option to remove pixels from the surrounding eye region. This ensures that these pixels are excluded from further analysis. The size of the eraser can be customized in the settings window.

.. figure:: _static/Pupil_roi_chosen.png
   :alt: Image Pupil_roi_chosen
   :width: 400px
   :align: center

   Example of eyeball chosen ROI.

.. raw:: html

   <div style="margin-bottom: 20px;"></div>


.. figure:: _static/erased_eye.png
   :alt: Image erased_eye
   :width: 300px
   :align: center

   Erasing pixels from the surrounding eye region.

.. raw:: html

   <div style="margin-bottom: 20px;"></div>

.. important::
    Don’t erase regions the pupil might cross.

Pupil Area Visualization Modes
------------------------------

FaceIt provides two ways to visualize the pupil area:

- **Normal preview** — continuous/raw pupil area trace.
- **Binary preview** — area estimated from a thresholded (binary) pupil mask.

Toggling the view
~~~~~~~~~~~~~~~~~

Use the **Show_binary** checkbox to switch between modes.

Binarization methods
--------------------

You can choose how the binary mask is created. Two methods are available:

- **Global (constant) binarization**
  Applies a single threshold to the whole image.

- **Adaptive binarization**
  Computes a local threshold per neighborhood (robust to uneven illumination).

Selecting the method
~~~~~~~~~~~~~~~~~~~~

By default, **Adaptive binarization** is used.

To switch methods, use the **Constant Binary** checkbox:

- **Unchecked (default)** → **Adaptive** binarization.
- **Checked** → **Global (constant)** binarization.

When **Constant Binary** is checked, a **threshold slider** becomes active so you can set the
global threshold used for the constant method.

.. note::
   - In **Adaptive** mode, the global threshold slider is disabled; instead, tune
     **Block size** and **C** under *Adaptive thresholding settings*.
   - In **Constant** mode, adjust the **threshold slider** to control the binary mask.

Parameters
~~~~~~~~~~

- **Global (constant)**:

  - **Binary threshold**: the global threshold value applied to all pixels.

- **Adaptive** (see **Adaptive thresholding settings**):

  - **Block size**: odd window size for local statistics (larger → smoother, less detail).
  - **C**: constant subtracted from the local mean/weighted mean (higher ``C`` → stricter threshold).

When to use which
~~~~~~~~~~~~~~~~~

- Use **Global** when lighting is uniform and the pupil/eyeball contrast is stable.
- Use **Adaptive** when lighting is uneven, there are vignetting, or contrast varies across the frame.

Tips
~~~~

- If the binary mask looks noisy or fragmented, try:

  - adjusting **Block size** (Adaptive)
  - adjusting **C** (Adaptive)

- The **Show_binary** checkbox controls **visualization**. The chosen **Binarization method** controls **how** the mask is computed.

.. _reflection-handling:

Reflection Correction
=====================

Bright corneal reflections can fragment the pupil mask and bias ellipse fitting. FaceIt
handles reflections in two ways:

- **Automatic detection + inpainting** (available **only** with *Adaptive* binarization; **default**)
- **Manual reflection ellipses** (available with *Adaptive* **and** *Constant/Global*)

Defaults
--------

- The default **Binarization method** is **Adaptive**.
- In **Adaptive** mode, the pipeline applies **automatic reflection detection + inpainting**
  unless you provide manual ellipses.

Behavior by binarization method
-------------------------------

+--------------------+---------------------------+-------------------------------+
| Thresholding mode  | Auto detect + inpaint     | Manual ellipses               |
+====================+===========================+===============================+
| **Adaptive**       | **Yes** (default)         | Inpaint using ellipses        |
+--------------------+---------------------------+-------------------------------+
| **Constant/Global**| **No**                    | Overlap fix (no inpainting)   |
+--------------------+---------------------------+-------------------------------+

How it works
------------

**Automatic (Adaptive only)**

1. Detect bright regions using a percentile threshold controlled by **Reflect br**;
   filter by area/circularity; dilate proportionally to glare size.
2. **Inpaint** the detected mask (TELEA) to remove glare before adaptive thresholding.

**Manual ellipses (both modes)**

- **Adaptive**: skip auto-detect and **inpaint** directly using the provided ellipses.
- **Constant/Global**: after thresholding/clustering, apply an **overlap fix**—pixels
  where the fitted **pupil ellipse** overlaps a **reflection ellipse** are restored to
  the pupil mask (no inpainting).

Controls & parameters
---------------------

- **Binarization method**
  - **Adaptive** (default): uses **Block size** (odd) and **C** (subtractive constant).
  - **Constant/Global**: uses one **Binary threshold**.
- **Reflect br** (slider, *Adaptive only*): sets the brightness percentile for
  automatic detection (higher → stricter, fewer pixels marked as reflections).
- **Manual reflection ellipses** (optional): user-specified ellipse masks used as above.



.. figure:: _static/reflection_added.png
   :alt: Image reflection_added
   :width: 300px
   :align: center

   Adding reflection cover to the pupil.

.. raw:: html

   <div style="margin-bottom: 20px;"></div>

Use the **Saturation Slider** to adjust the eyeball's saturation and achieve the optimal contrast between the pupil and the eyeball.
After completing the previous steps, use the frame slider to review the quality of pupil detection throughout your data and make any necessary adjustments. Once satisfied, select the **Pupil** checkbox and click **Process** to begin the analysis.
Once the analysis is complete, a pupil area plot will appear on the GUI. You can utilize the blinking detection button to detect and remove blinking, after which a new plot, excluding blinking, will be displayed on the GUI.

Analyse whisker pad
^^^^^^^^^^^^^^^^^^^

To analyse Whisker pad motion energy you can start by defining your region of interest using **Face ROI** bottom in the **ROI tools** section. check **whisker pad** checkbox and click on the process bottom.
After the analysis is complete, a whisker pad motion energy plot will be displayed on the GUI. If grooming activity is present in your data, you can easily interpolate the grooming segments by setting a threshold on the y-axis of the motion energy plot. To do this, click on **Define Grooming Threshold** and select the area where you want to remove activity above the specified level. A new plot, with the grooming segments interpolated, will then be displayed.

Saving data
^^^^^^^^^^^

When you click the save button, the processing results will automatically be stored in **.npz** files. To save the data in **.nwb** format, ensure you select the **Save NWB** checkbox before saving.