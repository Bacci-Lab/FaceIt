Pupil Processing
================

Overview
--------

Numerous studies have demonstrated the significance of pupillometry in examining brain states and cognitive functions.
By measuring pupil size variations, researchers can gain insights into mental states, cognitive load, and attentional processes,
making it a valuable tool in neuroscience research.

FaceIt offers an fast and precise approach for tracking pupil movement and size. Its user-friendly graphical interface makes it simple to use and facilitates effective data quality control.


This document describes the Face-IT pupil pipeline and its API
surface (key functions and parameters). For more information about Face-It please read the paper.

API Surface
-----------

.. py:method:: detect_pupil(frame_gray, erased_pixels, reflect_ellipse, mnd, reflect_brightness, clustering_method, binary_method, binary_threshold, c_value, block_size)

   Detect the pupil ellipse within a user-defined ROI, accounting for optional
   inpainting/removal of reflections and user-erased pixels.

   :param ndarray frame_gray: Grayscale ROI frame (after ROI cropping).
   :param ndarray erased_pixels: Boolean mask to exclude pixels (user eraser). Same shape as ROI.
   :param tuple reflect_ellipse: Optional reflection ROI(s) (center, axes, angle) if using manual overlap.
   :param int mnd: Minimum neighborhood distance / epsilon (DBSCAN) or contour filter scale.
   :param float reflect_brightness: Percentile or absolute cutoff used for reflection auto-detection (adaptive only).
   :param str clustering_method: ``"dbscan"`` or ``"contour"`` (simple contour-based).
   :param str binary_method: ``"adaptive"`` or ``"constant"``.
   :param float binary_threshold: Global threshold (if ``binary_method="constant"``).
   :param float c_value: Adaptive threshold offset (:math:`C`) (if ``binary_method="adaptive"``).
   :param int block_size: Adaptive neighborhood size (odd, e.g., 17) (if ``binary_method="adaptive"``).
   :returns: (ok, center_xy, width, height, angle_deg, diagnostics)
   :rtype: tuple

.. py:method:: preprocess_brightness(frame_gray, saturation_method, brightness, brightness_curve, secondary_BrightGain, brightness_concave_power, primary_direction, secondary_direction, saturation_ununiform)

   Apply optional uniform or gradual intensity/saturation adjustments in HSV-space
   before binarization.

.. py:method:: auto_reflection_mask(frame_gray, reflect_brightness)
.. py:method:: inpaint_reflections(frame_gray, reflection_mask)

.. py:method:: cluster_mask(binary, method="dbscan", mnd=5)
.. py:method:: fit_ellipse_from_mask(mask)

Light and Intensity Adjustment
------------------------------

Before segmentation, frames can be optionally preprocessed to enhance contrast and
stabilize binarization.

1) Uniform Intensity and Saturation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a spatially uniform correction (value channel scaling and optional saturation
scaling). Recommended when illumination is already fairly uniform.

- Pros: Simple, fast, stable.
- Cons: Does not fix lateral gradients or edge shadowing.

2) Gradual Intensity Adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compensates uneven illumination with **directional** and **concave** gradients:

- *Directional gradient:* increases brightness along one axis (left/right/up/down),
  with adjustable gain and curvature (``brightness_curve``).
- *Concave gradient:* symmetric brightening toward periphery to counter edge shadows
  (controlled by ``brightness_concave_power`` and ``secondary_BrightGain``).

Both operations modify the HSV **V** channel; saturation can be scaled uniformly or
proportionally (``saturation_ununiform``) to preserve contrast after brightening.

Image Binarization
------------------

Binarization converts the preprocessed grayscale ROI into a pupil candidate mask.

Adaptive Thresholding
~~~~~~~~~~~~~~~~~~~~~

For non-uniform illumination and dynamic contrast. For pixel :math:`(x, y)`:

.. math::

   T(x,y) = \mathrm{mean}_{N(x,y)} - C

Pixels with intensity lower than :math:`T(x,y)` are set to 1 (pupil candidates).
``block_size`` defines the neighborhood; ``c_value`` controls sensitivity.

- Pros: Robust to gradients and slow brightness drift.
- Cons: Slightly more compute; auto-reflection removal is supported here.

Constant (Global) Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a single threshold ``binary_threshold`` across the frame.

- Pros: Fast; stable when illumination is uniform.
- Cons: Sensitive to shadows; manual reflection handling recommended.

Light Reflection Handling
-------------------------

Strong corneal glints can break the pupil mask. Face-IT supports both **automatic**
and **manual/iterative** handling; the choice depends on the thresholding mode
and data quality.

1) Automatic (Adaptive Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Detect bright pixels by percentile (``reflect_brightness``) per-frame.
- Extract and filter contours by size/circularity to suppress large bright regions
  unrelated to glints.
- Dilate for coverage; **inpaint** selected regions to remove glints while preserving
  local texture.
- Proceed to adaptive thresholding on the inpainted frame.

2) Manual / Iterative (Adaptive or Constant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The user specifies reflection regions (e.g., ``reflect_ellipse``) or draws them.
- **Adaptive:** you may still prefer inpainting if glints are large or non-stationary.
- **Constant:** use the **iterative overlap** approach: fit a preliminary ellipse
  from the current mask, add only the overlap between reflection regions and the
  preliminary ellipse to the mask, refit the ellipse, and iterate a few rounds to
  converge.

Clustering
----------

After reflection handling and binarization, clustering isolates the pupil region.

DBSCAN
~~~~~~

Density-based clustering over foreground (non-zero) pixels with a minimal neighborhood
distance (``mnd``). Select the largest cluster; compute a **convex hull** to obtain
a contiguous mask for ellipse fitting.

- Effective when contours are fragmented due to glints or partial occlusion.
- ``mnd`` plays the role of epsilon; tune per video scale.

Simple Contour-Based
~~~~~~~~~~~~~~~~~~~~

Detect external contours; discard candidates that cover >~80% of width or have
extreme aspect ratios. Select the largest valid contour and use its **convex hull**
as the final mask.

- Faster and simpler; suitable when binarization already yields a clean blob.

Ellipse Fitting
---------------

Fit an ellipse to the non-zero pixels of the clustered mask using PCA over centered
coordinates. Let :math:`\mathbf{X} = \{(x_i, y_i)\}` be the pixel set; compute the
covariance:

.. math::

   \Sigma = \frac{1}{n-1}\sum_{i=1}^{n}
   \begin{bmatrix}
     x_i - \bar{x} \\
     y_i - \bar{y}
   \end{bmatrix}
   \begin{bmatrix}
     x_i - \bar{x} & y_i - \bar{y}
   \end{bmatrix}

Eigen-decompose:

.. math::

   \Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i,\quad i \in \{1,2\}

- Eigenvectors :math:`\mathbf{v}_1, \mathbf{v}_2` give ellipse orientation.
- Axis lengths scale with :math:`\sqrt{\lambda_1}` and :math:`\sqrt{\lambda_2}`.
- Report: center (mean), width (major), height (minor), and angle in degrees.

User-Erased Pixels
------------------

The GUI eraser sets marked pixels to zero in a persistent mask (``erased_pixels``)
that is applied **before** binarization and clustering. This is useful when the ROI
includes eyelids/fur borders or other persistent distractors.

.. important::
   Avoid erasing inside the pupil area itself.

Big Eye Movements (Saccade-like Events)
---------------------------------------

Large eye movements are estimated from frame-to-frame differences of the fitted
center coordinates along each axis (x, y). For each axis, compute the discrete
difference and suppress small changes (< 2 px) by replacing them with NaN.

.. py:method:: Saccade(pupil_center_i)

- Output is a per-frame displacement trace with small jitter removed.
- The first computed value is duplicated to preserve length consistency.

Blink Detection
---------------

Blinks are detected via changes in **pupil area** and **width/height ratio** using
a moving-variance approach with an adaptive threshold.

.. py:method:: detect_blinking_ids(pupil_data, threshold_factor, window_size)

.. py:method:: detect_blinking(pupil, width, height, x_saccade, y_saccade)

Method summary:

- Compute moving variance on pupil area and on width/height ratio.
- Derive a threshold from the range (max–min) of each moving-variance trace divided
  by ``threshold_factor``.
- Mark indices exceeding the threshold as blink candidates.
- Final blink indices are the union of the two strategies.
- Provide (a) indices of blinks and (b) a pupil trace with blinks excluded.

Diagnostics and Returns
-----------------------

``detect_pupil`` returns both geometry and optional diagnostics:

- **center_xy** (float, float): pupil center in ROI coordinates
- **width, height** (float): ellipse axes (pixels)
- **angle_deg** (float): rotation (degrees)
- **diagnostics** (dict): may include masks (binary/cluster), reflection masks,
  inpainting flags, thresholds used (``binary_threshold`` or adaptive ``c_value``),
  cluster stats, and warnings (e.g., small mask, degenerate covariance).

Recommendations
---------------

- Prefer **adaptive thresholding** when illumination varies; enable **auto reflection**
  for glint-heavy videos.
- Use **DBSCAN** when contours fragment; otherwise **contour-based** is fast and
  accurate.
- Tune ``block_size`` and ``c_value`` jointly; larger ROIs often benefit from
  slightly larger neighborhoods and a slightly higher |C|.
- Reserve **manual/iterative reflection** for constant-threshold workflows or when
  experimenter control is required.

