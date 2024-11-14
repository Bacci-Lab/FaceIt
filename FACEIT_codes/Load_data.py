import os
import numpy as np
import cv2
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PyQt5 import QtWidgets
from FACEIT_codes import functions
import threading
class LoadData:
    def __init__(self, app_instance):
        # Store a reference to the main app instance
        self.app_instance = app_instance

    def open_image_folder(self):
        """Open a folder dialog to select a directory containing images."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_path = os.path.join(project_root, "test_data", "test_images")
        print("Default path:", default_path)

        self.app_instance.folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.app_instance, "Select Folder", default_path
        )
        if self.app_instance.folder_path:
            self.app_instance.save_path = self.app_instance.folder_path
            npy_files = [f for f in os.listdir(self.app_instance.folder_path) if f.endswith('.npy')]
            self.app_instance.len_file = len(npy_files)
            self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
            self.app_instance.NPY = True
            self.app_instance.video = False
            self.display_graphics(self.app_instance.folder_path)
            self.app_instance.FaceROIButton.setEnabled(True)
            self.app_instance.PupilROIButton.setEnabled(True)

    def load_video(self):
        """Load video and prepare for processing."""
        self.app_instance.folder_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.app_instance, "Load Video", "", "Video Files (*.avi)"
        )
        if self.app_instance.folder_path:
            directory_path = os.path.dirname(self.app_instance.folder_path)
            self.app_instance.save_path = directory_path
            cap = cv2.VideoCapture(self.app_instance.folder_path)
            self.app_instance.len_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self.app_instance.Slider_frame.setMaximum(self.app_instance.len_file - 1)
            self.app_instance.video = True
            self.app_instance.NPY = False
            self.display_graphics(self.app_instance.folder_path)
            self.app_instance.FaceROIButton.setEnabled(True)
            self.app_instance.PupilROIButton.setEnabled(True)

    def load_image(self, filepath, image_height=384):
        """Load and resize a single image from the given file path."""
        try:
            current_image = np.load(filepath, allow_pickle=True)
            original_height, original_width = current_image.shape
            aspect_ratio = original_width / original_height
            image_width = int(image_height * aspect_ratio)
            resized_image = cv2.resize(current_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            return resized_image
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def load_images_from_directory(self, directory, image_height=384, max_workers=8):
        """Load images from a directory using multithreading."""
        file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        self.app_instance.progressBar.setMaximum(len(file_list))
        images = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.load_image, file, image_height): file for file in file_list}

            for i, future in enumerate(as_completed(futures)):
                image = future.result()
                if image is not None:
                    images.append(image)
                self.app_instance.progressBar.setValue(i + 1)

        self.app_instance.progressBar.setValue(len(file_list))
        return images

    def load_frames_from_video(self, video_path, max_workers=8, buffer_size=32, image_height=384):
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
            frame_queue.put(None)  # Sentinel to indicate end of frames

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

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        consumer()
        producer_thread.join()
        cap.release()

        return frames

    def display_graphics(self, folder_path):
        """Display initial graphics and setup scenes."""
        self.app_instance.frame = 0
        if self.app_instance.NPY:
            self.app_instance.image = functions.load_npy_by_index(folder_path, self.app_instance.frame)
        elif self.app_instance.video:
            self.app_instance.image = functions.load_frame_by_index(folder_path, self.app_instance.frame)

        functions.initialize_attributes(self.app_instance, self.app_instance.image)
        self.app_instance.scene2 = functions.second_region(
            self.app_instance.graphicsView_subImage,
            self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width,
            self.app_instance.image_height
        )
        self.app_instance.graphicsView_MainFig, self.app_instance.scene = functions.display_region(
            self.app_instance.image,
            self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width,
            self.app_instance.image_height
        )
