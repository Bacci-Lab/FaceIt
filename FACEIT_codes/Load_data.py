import os
import numpy as np
import cv2
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PyQt5 import QtWidgets
from FACEIT_codes import functions
from line_profiler import profile
import threading
class LoadData:
    def __init__(self, app_instance):
        # Store a reference to the main app instance
        self.app_instance = app_instance
    def open_image_folder(self):
        """Open a folder dialog to select a directory containing .npy image files."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_path = os.path.join(project_root, "test_data", "test_images")

        # Open folder selection dialog
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.app_instance, "Select Folder", default_path)

        if not folder_path:
            return

        self.app_instance.folder_path = folder_path
        self.app_instance.save_path = folder_path

        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]


        # Configure application settings if valid .npy files are found
        self.app_instance.len_file = len(npy_files)
        self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
        self.app_instance.NPY = True
        self.app_instance.video = False
        self.display_graphics(folder_path)

        # Enable buttons after successful file check
        self.app_instance.FaceROIButton.setEnabled(True)
        self.app_instance.PupilROIButton.setEnabled(True)
        self.app_instance.Slider_frame.setEnabled(True)

    def load_video(self):
        """Load video and prepare for processing."""
        self.app_instance.folder_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.app_instance, "Load Video", "", "Video Files (*.avi *.wmv *mp4)"
        )
        if self.app_instance.folder_path:
            directory_path = os.path.dirname(self.app_instance.folder_path)
            self.app_instance.save_path = directory_path
            self.app_instance.cap = cv2.VideoCapture(self.app_instance.folder_path)
            self.app_instance.len_file = int(self.app_instance.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
            self.app_instance.video = True
            self.app_instance.NPY = False
            self.display_graphics(self.app_instance.folder_path)
            self.app_instance.FaceROIButton.setEnabled(True)
            self.app_instance.PupilROIButton.setEnabled(True)
            self.app_instance.Slider_frame.setEnabled(True)

    def load_image(self, filepath, image_height):
        """Load and resize a single image from the given file path."""
        try:
            current_image = np.load(filepath, allow_pickle=True)
            height, width = current_image.shape[:2]
            original_height, original_width = current_image.shape
            aspect_ratio = original_width / original_height
            image_width = int(image_height * aspect_ratio)
            resized_image = cv2.resize(current_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            return resized_image
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def load_images_from_directory(self, directory, image_height, max_workers=8):
        """Load images from a directory using multithreading while preserving order and improving performance."""
        start_time = time.time()
        file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        images = [None] * len(file_list)  # Preallocate the list to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_image, file, image_height): i
                for i, file in enumerate(file_list)
            }

            with tqdm(total=len(file_list), desc="Loading .npy images") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    image = future.result()
                    if image is not None:
                        images[index] = image
                    pbar.update(1)

        elapsed_time = time.time() - start_time
        print(f"✅ Image loading completed in {elapsed_time:.2f} seconds.")
        return images

    def load_frames_from_video(self, video_path,image_height, max_workers=8, buffer_size=32):
        """Load frames from a video file using multithreading."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_queue = queue.Queue(maxsize=buffer_size)

        def producer():
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)
            frame_queue.put(None)

        def resize_frame(frame, image_height):
            original_height, original_width, _ = frame.shape if len(frame.shape) == 3 else (frame.shape[0], frame.shape[1])
            aspect_ratio = original_width / original_height
            image_width = int(image_height * aspect_ratio)
            return cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)

        def consumer():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                with tqdm(total=total_frames, desc="Processing video frames") as pbar:
                    while True:
                        frame = frame_queue.get()
                        if frame is None:
                            break
                        future = executor.submit(resize_frame, frame, image_height)
                        futures.append(future)

                        for future in as_completed(futures):
                            frames.append(future.result())
                            pbar.update(1)
                            futures.remove(future)


        start_time = time.time()
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        consumer()
        producer_thread.join()
        cap.release()
        elapsed_time = time.time() - start_time
        print(f"✅ Video frame loading completed in {elapsed_time:.2f} seconds.")

        return frames

    def display_graphics(self, folder_path):
        """Display initial graphics and setup scenes."""
        self.app_instance.frame = 0
        if self.app_instance.NPY:
            self.app_instance.image = functions.load_npy_by_index(folder_path, self.app_instance.frame)
        elif self.app_instance.video:
            self.app_instance.image = functions.load_frame_by_index(self.app_instance.cap, self.app_instance.frame)

        functions.initialize_attributes(self.app_instance, self.app_instance.image)
        self.app_instance.scene2 = functions.second_region(
            self.app_instance.graphicsView_subImage,
            self.app_instance.graphicsView_MainFig,
        )
        self.app_instance.graphicsView_MainFig, self.app_instance.scene = functions.display_region(
            self.app_instance.image,
            self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width,
            self.app_instance.image_height
        )
