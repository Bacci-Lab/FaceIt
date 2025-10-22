from PyQt5.QtCore import QThread
from FACEIT_codes import pupil_detection
from FACEIT_codes.Workers import MotionWorker
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os, time, math, logging
import numpy as np
import cv2
import logging
from multiprocessing import cpu_count  # keep if you still use elsewhere
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    cv2.setNumThreads(0)
except Exception:
    pass

from FACEIT_codes.functions import (
    change_saturation_uniform,
    change_Gradual_saturation,
    apply_intensity_gradient_gray,
    SaturationSettings, show_ROI2
)

def _fmt_secs(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def process_single_frame(args):
    (
        current_image, sub_image,
        saturation, contrast,
        erased_pixels, reflect_ellipse,
        eye_corner_center, mnd,
        reflect_brightness,
        clustering_method, binary_method, binary_threshold,
        saturation_method,
        brightness, brightness_curve,
        secondary_BrightGain, brightness_concave_power,
        saturation_ununiform,
        primary_direction, secondary_direction,
        c_value, block_size,
        disable_filtering,                      # ← NEW: bool passed in args
    ) = args

    sub_region, frame_pos, pupil_frame_center, frame_axes = show_ROI2(sub_image, current_image)

    settings = SaturationSettings(
        primary_direction=primary_direction,
        brightness_curve=brightness_curve,
        brightness=brightness,
        secondary_direction=secondary_direction,
        brightness_concave_power=brightness_concave_power,
        secondary_BrightGain=secondary_BrightGain,
        saturation_ununiform=saturation_ununiform,
    )

    # Gradual / Uniform processing (unchanged) ...
    if saturation_method == "Gradual":
        if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
            processed = apply_intensity_gradient_gray(sub_region, settings)
        else:
            if (np.allclose(sub_region[..., 0], sub_region[..., 1]) and
                np.allclose(sub_region[..., 1], sub_region[..., 2])):
                gray = cv2.cvtColor(sub_region, cv2.COLOR_BGR2GRAY)
                processed = apply_intensity_gradient_gray(gray, settings)
            else:
                processed = change_Gradual_saturation(sub_region, settings)
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    elif saturation_method == "Uniform":
        if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
            sub_region = cv2.cvtColor(sub_region, cv2.COLOR_GRAY2BGR)
        processed = change_saturation_uniform(sub_region, saturation=saturation, contrast=contrast)
    else:
        processed = sub_region.copy()

    # Detector expects BGRA
    sub_region_rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)

    _, center, width, height, angle, current_area = pupil_detection.detect_pupil(
        sub_region_rgba,
        erased_pixels, reflect_ellipse, mnd, reflect_brightness,
        clustering_method, binary_method, binary_threshold, c_value, block_size,
        disable_filtering=disable_filtering,          # ← use the bool you unpacked
    )

    pupil_distance_from_corner = (
        math.hypot(center[0] - eye_corner_center[0], center[1] - eye_corner_center[1])
        if eye_corner_center is not None else np.nan
    )

    center = (int(center[0]), int(center[1]))
    width, height = int(width), int(height)
    current_area = np.pi * width * height

    return (current_area, center, center[0], center[1],
            width, height, pupil_distance_from_corner,
            frame_pos, pupil_frame_center, frame_axes)




def display_show_ROI(ROI, image):

    if isinstance(ROI, tuple):
        x, y, width, height = ROI
        sub_image = image[int(y):int(y + height), int(x):int(x + width)]
        return sub_image, ROI
    sub_image = ROI.rect()
    return sub_image, ROI


def create_roi(x, y, width, height):
    """Creates a representation of a region of interest (ROI) based on given coordinates and size."""
    # Return a simple dictionary or tuple with ROI properties
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height
    }
def _make_roi_slicer(sub_image, first_frame):
    if isinstance(sub_image, tuple):
        x, y, w, h = map(int, sub_image)
        return lambda frame: frame[y:y+h, x:x+w]
    else:
        sub_region, ROI_pos, pupil_frame_center, frame_axes = show_ROI2(sub_image, first_frame)
        if isinstance(ROI_pos, tuple):
            x, y, w, h = map(int, ROI_pos)
            return lambda frame: frame[y:y+h, x:x+w]
        return lambda frame: show_ROI2(sub_image, frame)[0]
def _detect_frame(i, frame, roi_slice, cfg):
    # slice ROI (zero-copy)
    roi = roi_slice(frame)

    # saturation / brightness
    processed, is_gray = _apply_saturation_fast(
        roi,
        cfg["saturation_method"],
        cfg["settings"],
        cfg["saturation"],
        cfg["contrast"]
    )

    # choose detector input format
    if cfg["needs_bgra"]:
        if is_gray:
            bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            img_for_det = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        else:
            img_for_det = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)
    else:
        img_for_det = processed  # pass GRAY or BGR

    _, center, width, height, angle, _ = pupil_detection.detect_pupil(
        img_for_det,
        cfg["erased_pixels"], cfg["reflect_ellipse"], cfg["mnd"],
        cfg["reflect_brightness"], cfg["clustering_method"], cfg["binary_method"],
        cfg["binary_threshold"], cfg["c_value"], cfg["block_size"],
        disable_filtering=cfg["disable_filtering"],
    )

    cx, cy = int(center[0]), int(center[1])
    w, h = int(width), int(height)
    area = math.pi * w * h

    if cfg["eye_corner_center"] is not None:
        ex, ey = cfg["eye_corner_center"]
        dist = math.hypot(cx - ex, cy - ey)
    else:
        dist = float("nan")

    return (i, area, cx, cy, w, h, dist, angle)
def _apply_saturation_fast(bgr_roi, saturation_method, settings, saturation, contrast):
    if saturation_method == "Gradual":
        if bgr_roi.ndim == 2 or (
            bgr_roi.ndim == 3 and
            np.allclose(bgr_roi[...,0], bgr_roi[...,1]) and
            np.allclose(bgr_roi[...,1], bgr_roi[...,2])
        ):
            gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY) if bgr_roi.ndim == 3 else bgr_roi
            return apply_intensity_gradient_gray(gray, settings), True
        else:
            out = change_Gradual_saturation(bgr_roi, settings)
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), False
    elif saturation_method == "Uniform":
        if bgr_roi.ndim == 2:
            bgr_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_GRAY2BGR)
        return change_saturation_uniform(bgr_roi, saturation=saturation, contrast=contrast), False
    return bgr_roi, (bgr_roi.ndim == 2)



def _detect_one(i, images, roi_slice, cfg):
    frame = images[i]
    roi = roi_slice(frame)

    processed, is_gray = _apply_saturation_fast(
        roi,
        cfg["saturation_method"],
        cfg["settings"],
        cfg["saturation"],
        cfg["contrast"]
    )

    if cfg["needs_bgra"]:
        if is_gray:
            bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            img_for_det = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        else:
            img_for_det = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)
    else:
        img_for_det = processed  # pass GRAY or BGR as your detector supports

    _, center, width, height, angle, _ = pupil_detection.detect_pupil(
        img_for_det,
        cfg["erased_pixels"], cfg["reflect_ellipse"], cfg["mnd"],
        cfg["reflect_brightness"], cfg["clustering_method"], cfg["binary_method"],
        cfg["binary_threshold"], cfg["c_value"], cfg["block_size"],
        disable_filtering=cfg["disable_filtering"],
    )

    cx, cy = int(center[0]), int(center[1])
    width, height = int(width), int(height)
    area = math.pi * width * height

    if cfg["eye_corner_center"] is not None:
        ex, ey = cfg["eye_corner_center"]
        dist = math.hypot(cx - ex, cy - ey)
    else:
        dist = float("nan")

    return (area, (cx, cy), cx, cy, width, height, dist, cfg["pupil_frame_center"])

class ProcessHandler:
    def __init__(self, app_instance):
        # Store a reference to the app instance for accessing its attributes and methods
        self.app_instance = app_instance


    def process(self):
        """Processes pupil and face data if selected and images are loaded."""
        # Process pupil data if the pupil checkbox is checked
        if self.app_instance.pupil_check():
            self.process_pupil_data()

        # Process face data if the face checkbox is checked
        if self.app_instance.face_check():
            self.process_face_data()

    def process_pupil_data(self):
        """Processes pupil data asynchronously using a worker thread."""
        if not self.app_instance.Pupil_ROI_exist:
            self.app_instance.warning("NO Pupil ROI is chosen!")
            return

        self.load_images_if_needed()
        self.app_instance.start_pupil_dilation_computation(self.app_instance.images)

    def process_face_data(self):
        if not self.app_instance.Face_ROI_exist:
            self.app_instance.warning("NO Face ROI is chosen!")
            return

        # --- Make sure frames are loaded at least once ---
        if not hasattr(self.app_instance, "images") or self.app_instance.images is None:
            self.load_images_if_needed()

        # --- Load frames only now (when Process is pressed) ---
        if self.app_instance.video:
            # fresh generator each time Process is pressed
            loader = self.app_instance.load_handler
            images = loader.load_frames_from_video(
                self.app_instance.folder_path,
                self.app_instance.image_height
            )
        elif self.app_instance.NPY:
            # .npy mode: already preloaded into memory
            images = self.app_instance.images
        else:
            self.app_instance.warning("No video or image source loaded.")
            return

        # --- Start motion analysis ---
        self.motion_thread = QThread()
        self.motion_worker = MotionWorker(images, self.app_instance.Face_frame)
        self.motion_worker.moveToThread(self.motion_thread)

        # Signals
        self.motion_thread.started.connect(self.motion_worker.run)
        self.motion_worker.finished.connect(self.handle_motion_result)
        self.motion_worker.error.connect(self.handle_worker_error)

        # Cleanup
        self.motion_worker.finished.connect(self.motion_thread.quit)
        self.motion_worker.finished.connect(self.motion_worker.deleteLater)
        self.motion_thread.finished.connect(self.motion_thread.deleteLater)
        self.motion_thread.start()

    def handle_motion_result(self, result):
        self.app_instance.motion_energy = result
        if result is None or len(result) == 0:
            print("[DEBUG] Motion result is empty – skipping plot")
            return
        self.app_instance.plot_handler.plot_result(
            result,
            self.app_instance.graphicsView_whisker,
            "motion"
        )

    def handle_worker_error(self, error_msg):
        self.app_instance.warning(f"Motion processing error: {error_msg}")

    def load_images_if_needed(self):
        """Loads images if they are not already loaded."""
        if not self.app_instance.Image_loaded:
            if self.app_instance.NPY == True:
                # Load images from a directory of .npy files
                self.app_instance.images = self.app_instance.load_handler.load_images_from_directory(self.app_instance.folder_path,self.app_instance.image_height)
            elif self.app_instance.video == True:
                # Load images from a video file
                self.app_instance.images = self.app_instance.load_handler.load_frames_from_video(self.app_instance.folder_path, self.app_instance.image_height)

            # Mark images as loaded
            self.app_instance.Image_loaded = True

    def detect_blinking(self, pupil, width, height, x_saccade, y_saccade):
        """
        Detects and processes blinking events based on pupil data, width, and height ratios.

        Parameters:
            pupil (array): Array of pupil data.
            width (array): Array of pupil width data.
            height (array): Array of pupil height data.
            x_saccade (array): Array of X-axis saccade data.
            y_saccade (array): Array of Y-axis saccade data.
        """

        # Convert width, height, and saccade data to numpy arrays
        width = np.array(width)
        height = np.array(height)
        self.app_instance.X_saccade_updated = np.array(x_saccade)
        self.app_instance.Y_saccade_updated = np.array(y_saccade)

        # Calculate the width-to-height ratio to identify potential blinking
        ratio = width / height

        # Detect blinking based on the ratio and pupil data using defined thresholds
        blinking_id_ratio = pupil_detection.detect_blinking_ids(ratio, 20)
        blinking_id_area = pupil_detection.detect_blinking_ids(pupil, 10)

        # Combine the detected blinking indices and remove duplicates
        combined_blinking_ids = list(set(blinking_id_ratio + blinking_id_area))

        # Ensure indices do not exceed the length of the saccade data
        combined_blinking_ids = [x for x in combined_blinking_ids if x < len(self.app_instance.X_saccade_updated[0])]
        combined_blinking_ids.sort()

        # Replace the detected blinking indices with NaNs in the updated saccade arrays
        self.app_instance.X_saccade_updated[0][combined_blinking_ids] = np.nan
        self.app_instance.Y_saccade_updated[0][combined_blinking_ids] = np.nan

        # Interpolate the pupil data to handle the blinking periods smoothly
        self.app_instance.interpolated_pupil = pupil_detection.interpolate(combined_blinking_ids, pupil)

        # Plot the interpolated pupil data with the detected saccade events
        self.app_instance.plot_handler.plot_result(
            self.app_instance.interpolated_pupil,
            self.app_instance.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.app_instance.X_saccade
        )

        # Store the final pupil area as the interpolated result
        self.app_instance.final_pupil_area = np.array(self.app_instance.interpolated_pupil)
        return combined_blinking_ids

    def Pupil_Filtering(self,
                        pupil,
                        x_saccade,
                        y_saccade,
                        win=15,
                        k=3.0):
        """
        Detect blinks by flagging outliers in the pupil signal via a Hampel filter
        (rolling median ± k * scaled‐MAD), then mask and interpolate saccade data.

        Parameters:
            pupil      (array): 1D pupil-area (or diameter) time series
            width      (array): [unused]—kept for signature compatibility
            height     (array): [unused]—kept for signature compatibility
            x_saccade  (array): raw X‐saccade time series
            y_saccade  (array): raw Y‐saccade time series
            win        (int): half‐window length for rolling stats (default 25)
            k          (float): outlier threshold multiplier (default 3.0)
        Returns:
            combined_blinking_ids (list): indices of samples flagged as blinks
        """

        # 1) Copy inputs into instance attributes
        pupil = np.asarray(pupil)
        self.app_instance.X_saccade_updated = np.asarray(x_saccade)
        self.app_instance.Y_saccade_updated = np.asarray(y_saccade)

        # 2) Set up Hampel parameters
        L = 1.4826  # consistency factor → MAD ≈ σ for Gaussians
        n = len(pupil)
        medians = np.zeros(n)
        mads = np.zeros(n)
        outliers = np.zeros(n, dtype=bool)

        # 3) Rolling‐window Hampel filter
        half = win // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)  # +1 because Python slicing is exclusive
            window = pupil[start:end]

            med = np.median(window)
            mad = L * np.median(np.abs(window - med))

            medians[i] = med
            mads[i] = mad

            # Flag as blink if it exceeds k × local MAD
            if mad > 0 and np.abs(pupil[i] - med) > k * mad:
                outliers[i] = True

        # 4) Extract the blink indices
        blinking_ids = np.where(outliers)[0].tolist()

        # 5) Mask saccade samples at those times
        max_idx = self.app_instance.X_saccade_updated.shape[-1]
        valid_ids = [i for i in blinking_ids if i < max_idx]
        valid_ids.sort()

        self.app_instance.X_saccade_updated[0, valid_ids] = np.nan
        self.app_instance.Y_saccade_updated[0, valid_ids] = np.nan

        # 6) Interpolate pupil through blinks
        self.app_instance.interpolated_pupil = pupil_detection.interpolate(valid_ids, pupil)

        # 7) Plot final result
        self.app_instance.plot_handler.plot_result(
            self.app_instance.interpolated_pupil,
            self.app_instance.graphicsView_pupil,
            "pupil",
            color="palegreen",
            saccade=self.app_instance.X_saccade
        )

        # 8) Store and return
        self.app_instance.final_pupil_area = np.array(self.app_instance.interpolated_pupil)
        return valid_ids

    def pupil_dilation_comput(self, images_iter, saturation, contrast, erased_pixels, reflect_ellipse,
                              mnd, reflect_brightness, clustering_method, binary_method, binary_threshold,
                              saturation_method, brightness, brightness_curve,
                              secondary_BrightGain, brightness_concave_power,
                              saturation_ununiform, primary_direction, secondary_direction,
                              c_value, block_size, sub_image):
        """
        Fast, bounded-thread streaming compute with 10% progress logs (single 0% line).
        """
        t0 = time.perf_counter()
        verbose = getattr(self.app_instance, "verbose", False)

        # ---- 1) Build a streaming iterator and get N if known ----
        if isinstance(images_iter, np.ndarray):
            images = images_iter
            n = int(images.shape[0])
            # first frame
            i0, first_frame = 0, images[0]
            # the rest
            frame_iter = ((i, images[i]) for i in range(1, n))
        else:
            n = int(getattr(self.app_instance, "len_file", 0)) or 0
            frame_iter = enumerate(images_iter)
            try:
                i0, first_frame = next(frame_iter)  # peek one
            except StopIteration:
                return tuple([np.array([])] * 13)  # empty

        if n == 0:
            return tuple([np.array([])] * 13)

        # ---- 2) ROI slicer and frame-invariant geometry from first frame ----
        roi_slice = _make_roi_slicer(sub_image, first_frame)
        _, frame_pos, pupil_frame_center, frame_axes = show_ROI2(sub_image, first_frame)

        # ---- 3) Settings/config (once) ----
        settings = SaturationSettings(
            primary_direction=primary_direction,
            brightness_curve=brightness_curve,
            brightness=brightness,
            secondary_direction=secondary_direction,
            brightness_concave_power=brightness_concave_power,
            secondary_BrightGain=secondary_BrightGain,
            saturation_ununiform=saturation_ununiform
        )
        needs_bgra = True  # set False if your detector accepts BGR/GRAY

        eye_corner_center = (
            (self.app_instance.eye_corner_center[0], self.app_instance.eye_corner_center[1])
            if getattr(self.app_instance, "eye_corner_center", None) is not None else None
        )

        cfg = dict(
            saturation_method=saturation_method,
            settings=settings,
            saturation=saturation,
            contrast=contrast,
            erased_pixels=erased_pixels,
            reflect_ellipse=reflect_ellipse,
            mnd=mnd,
            reflect_brightness=reflect_brightness,
            clustering_method=clustering_method,
            binary_method=binary_method,
            binary_threshold=binary_threshold,
            c_value=c_value,
            block_size=block_size,
            needs_bgra=needs_bgra,
            eye_corner_center=eye_corner_center,

            # NEW: read the checkbox from the main window; default False if missing
            disable_filtering=bool(getattr(self.app_instance, "deactivate_filtering", lambda: False)()),
        )

        # ---- 4) Preallocate outputs ----
        pupil_dilation = np.empty(n, dtype=np.float32)
        pupil_center_X = np.empty(n, dtype=np.int32)
        pupil_center_y = np.empty(n, dtype=np.int32)
        pupil_center = np.empty((n, 2), dtype=np.int32)
        pupil_width = np.empty(n, dtype=np.int32)
        pupil_height = np.empty(n, dtype=np.int32)
        pupil_distance = np.empty(n, dtype=np.float32)
        angle = np.empty(n, dtype=np.float32)

        # ---- 5) Thread pool with bounded in-flight futures ----
        workers = min(4, os.cpu_count() or 4)
        max_inflight = workers * 4
        if verbose:
            logging.info(f"[FaceIt] Threaded pupil compute with {workers} workers for {n} frames")

        start_loop = time.perf_counter()
        done = 0

        # Print 0% ONCE (before loop), then lock progress to 10% buckets
        logging.info(f"[FaceIt] 0% (0/{n}) | elapsed 00:00 | ETA --:--")
        if hasattr(self.app_instance, "progress_signal"):
            self.app_instance.progress_signal.emit(0, 0, n, 0.0, 0.0)
        last_bucket = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            inflight = set()

            # submit first frame immediately
            inflight.add(ex.submit(_detect_frame, i0, first_frame, roi_slice, cfg))

            # prime the pump
            while len(inflight) < min(max_inflight, max(1, n - 1)):
                try:
                    i, frame = next(frame_iter)
                except StopIteration:
                    break
                inflight.add(ex.submit(_detect_frame, i, frame, roi_slice, cfg))

            # consume and refill
            while inflight:
                done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)

                for fut in done_set:
                    (i, area, cx, cy, w, h, dist, fangle) = fut.result()

                    pupil_dilation[i] = area
                    pupil_center[i, 0] = cx
                    pupil_center[i, 1] = cy
                    pupil_center_X[i] = cx
                    pupil_center_y[i] = cy
                    pupil_width[i] = w
                    pupil_height[i] = h
                    pupil_distance[i] = dist
                    angle[i] = fangle

                    # progress every 10%
                    done += 1
                    if n:  # avoid div-by-zero
                        pct = int((done * 100) // n)  # 0..100
                        bucket = max(0, min(100, (pct // 10) * 10))  # 0,10,20,...,100
                        if bucket > last_bucket:
                            last_bucket = bucket
                            elapsed = time.perf_counter() - start_loop
                            fps = (done / elapsed) if elapsed > 0 else 0.0
                            eta = ((n - done) / fps) if fps > 0 else 0.0
                            logging.info(
                                f"[FaceIt] {bucket}% ({done}/{n}) | {fps:.1f} fps | "
                                f"elapsed {_fmt_secs(elapsed)} | ETA {_fmt_secs(eta)}"
                            )
                            if hasattr(self.app_instance, "progress_signal"):
                                self.app_instance.progress_signal.emit(int(bucket), done, n, float(fps), float(eta))

                while len(inflight) < max_inflight:
                    try:
                        i, frame = next(frame_iter)
                    except StopIteration:
                        break
                    inflight.add(ex.submit(_detect_frame, i, frame, roi_slice, cfg))

        # ---- 6) Saccades ----
        X_saccade = self.Saccade(pupil_center_X)
        Y_saccade = self.Saccade(pupil_center_y)

        if verbose:
            logging.info(f"[FaceIt] pupil_dilation_comput done in {time.perf_counter() - t0:.2f}s")

        return (pupil_dilation, pupil_center_X, pupil_center_y, pupil_center,
                X_saccade, Y_saccade, pupil_distance,
                pupil_width, pupil_height,  angle)

    def Saccade(self, pupil_center_i):
        """
        Computes the saccade movements based on changes in the pupil center coordinates.

        Parameters:
            pupil_center_i (array-like): The center coordinates of the pupil for each frame in the i axis.

        Returns:
            np.ndarray: A 2D array with saccade values, where small changes (<2) are replaced with NaN.
        """
        # Compute saccade differences using numpy's vectorized operations
        saccade = np.diff(pupil_center_i).astype(float)

        # Duplicate the first computed value at the beginning of the list to avoid changing dataset length
        if saccade.size > 0:
            first_value = saccade[0]
            saccade = np.insert(saccade, 0, first_value)


        # Set small movements (absolute value < 2) to NaN for better filtering of significant movements
        saccade[abs(saccade) < 2] = np.nan


        # Reshape the saccade array to be 2D (1 row) for consistency
        saccade = saccade.reshape(1, -1)


        #self.plot_saccade(pupil_center_i, saccade)



        return saccade

    def remove_grooming(self, grooming_thr, facemotion):
        """
        Caps the 'facemotion' data at a specified grooming threshold.

        Parameters:
        grooming_thr (float): The threshold above which 'facemotion' values are capped.
        facemotion (array-like): The array of facial motion data to be processed.

        Returns:
        tuple: A tuple containing:
            - facemotion_without_grooming (np.ndarray): The 'facemotion' array with values capped at the threshold.
            - grooming_ids (np.ndarray): Indices of elements in 'facemotion' that were capped.
            - grooming_thr (float): The grooming threshold used for capping.
        """
        # Ensure 'facemotion' is converted to a NumPy array for array operations.
        facemotion = np.asarray(facemotion)

        # Identify indices where 'facemotion' exceeds the grooming threshold.
        grooming_ids = np.where(facemotion >= grooming_thr)

        # Cap the 'facemotion' values at the threshold.
        facemotion_without_grooming = np.minimum(facemotion, grooming_thr)

        # Store the modified array as an attribute in the app instance.
        self.app_instance.facemotion_without_grooming = facemotion_without_grooming

        # Return the modified array, indices of capped values, and the threshold.
        return facemotion_without_grooming, grooming_ids, grooming_thr




