import numpy as np
import cv2
import matplotlib.pyplot as plt
from FACEIT_codes.functions import SaturationSettings, apply_intensity_gradient_gray
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QSizePolicy
from FACEIT_codes import functions
from FACEIT_codes import pupil_detection
from FACEIT_codes.functions import change_saturation_uniform, change_Gradual_saturation


class PlotHandler:
    def __init__(self,app_instance):
        self.app_instance = app_instance
        self.panning = False
        self.press_event = None

        # Create figure and canvas ONCE (for performance optimization)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)


    def plot_result(
            self,
            data: np.ndarray,
            graphics_view: QtWidgets.QGraphicsView,
            label: str,
            color: str = '#D97A53',
            saccade: np.ndarray = None,
            background_color: str = '#3d4242',
            grid: bool = False,
            legend_fontsize: int = 8,
            Cursor=False
    ):
        """
        Plots the data on a given graphics view using Matplotlib.

        Parameters:
        - data (np.ndarray): The data array to be plotted.
        - graphics_view (QtWidgets.QGraphicsView): The graphics view to display the plot.
        - label (str): The label for the plotted line.
        - color (str, optional): The color of the plot line. Default is '#D97A53'.
        - saccade (np.ndarray, optional): Optional saccade data to be plotted as a colormap.
        - background_color (str, optional): Background color of the plot. Default is '#3d4242'.
        - grid (bool, optional): Whether to show gridlines. Default is False.
        - legend_fontsize (int, optional): Font size of the legend. Default is 8.
        """
        if data is None or len(data) == 0:
            raise ValueError("Data for plotting cannot be None or empty.")
        self.app_instance.Save_Button.setEnabled(True)

        # Clear the graphics view and set up the canvas
        self._clear_graphics_view(graphics_view)

        # Create figure and axes
        fig, ax = plt.subplots()
        self._plot_data(ax, data, label, color, self.app_instance.frame)
        self._plot_saccade(ax, saccade, data)
        # Customize the plot
        self._customize_plot(fig, ax, data, background_color, grid, legend_fontsize)

        # Integrate the plot into the graphics view
        self._integrate_canvas_into_view(graphics_view, fig)
        if Cursor:
            self._plot_Cursor(ax, data, self.app_instance.frame)

    def _plot_Cursor(self, ax: plt.Axes, data, frame_index):
        # If a specific frame is provided, plot a marker at that frame
        if hasattr(self, 'vertical_line') and self.vertical_line is not None:
            self.vertical_line.remove()  # Removes the old line from the plot
            self.vertical_line = None  # Reset reference
            if frame_index is not None and 0 <= frame_index < len(data):
                self.vertical_line = ax.axvline(x=frame_index, color='blue', linewidth=2)


    def _plot_data(self, ax: plt.Axes, data: np.ndarray, label: str, color: str, frame_index: int = None):
        """Plots the main data on the provided axes, with an optional point marker at a specific frame."""

        x_values = np.arange(len(data))
        ax.plot(x_values, data, color=color, label=label, linestyle='--')

        if frame_index is not None and 0 <= frame_index < len(data):
            self.vertical_line = ax.axvline(x=frame_index, color='blue', linewidth=2)


    def _plot_saccade(self, ax: plt.Axes, saccade: np.ndarray, data: np.ndarray):
        """Plots the saccade data as a colormap if provided."""
        if saccade is not None:
            if saccade.shape[1] == len(data):
                saccade = saccade[:, 1:]  # Trim the first column to match the length

            data_max = np.max(data)
            range_val = np.max(data) - np.min(data)
            y_min = data_max + range_val / 10
            y_max = data_max + range_val / 5
            x_values = np.arange(len(data))
            ax.pcolormesh(x_values, [y_min, y_max], saccade, cmap='RdYlGn', shading='flat')
    def _customize_plot(
            self,
            fig: plt.Figure,
            ax: plt.Axes,
            data: np.ndarray,
            background_color: str,
            grid: bool,
            legend_fontsize: int
    ):
        """Customizes the appearance of the plot."""
        data_min, data_max = np.min(data), np.max(data)
        range_val = data_max - data_min

        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.set_yticks([])
        ax.set_xlim(0, len(data))
        ax.set_ylim(data_min, data_max + range_val / 4)

        # Customize axes and ticks
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom']:
            ax.spines[spine].set_visible(True)
        ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)

        # Add legend and set font color
        legend = ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)
        for text in legend.get_texts():
            text.set_color("white")
        ax.grid(grid)
        # Add zoom and pan interactions
        self._setup_interaction_events(fig, ax)

    def _integrate_canvas_into_view(self, graphics_view: QtWidgets.QGraphicsView, fig: plt.Figure):
        """Integrates the Matplotlib canvas into the provided PyQt graphics view."""
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()

        layout = QtWidgets.QVBoxLayout(graphics_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas)
        graphics_view.setLayout(layout)

        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0.01, right=1, top=0.95, bottom=0.25)
        canvas.draw()

    def _clear_graphics_view(self, graphics_view: QtWidgets.QGraphicsView):
        """Clears the given graphics view layout."""
        if graphics_view.layout() is not None:
            old_layout = graphics_view.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old_layout)

    def _setup_interaction_events(self, fig, ax):
        """Sets up basic zoom and pan events for the plot."""
        self.panning = False
        self.press_event = None

        def on_press(event):
            if event.inaxes != ax:
                return
            self.panning = True
            self.press_event = event
            event.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))

        def on_motion(event):
            if not self.panning or self.press_event is None or event.xdata is None:
                return
            dx = event.xdata - self.press_event.xdata
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            fig.canvas.draw_idle()

        def on_release(event):
            self.panning = False
            self.press_event = None
            event.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        def on_click(event):
            """Handles click events for setting grooming threshold."""
            if self.app_instance.find_grooming_threshold:
                if event.inaxes == ax and event.ydata is not None:
                    self.app_instance.grooming_thr = event.ydata
                    self.app_instance.find_grooming_threshold = False
                    self.app_instance.lineEdit_grooming_y.setText(str(int(event.ydata)))
                    self.app_instance.display_removed_grooming(self.app_instance.grooming_thr, self.app_instance.motion_energy)

        def zoom(event):
            """Handles zooming based on mouse scroll events."""
            current_xlim = ax.get_xlim()
            xdata = event.xdata
            if xdata is None:
                return
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            new_xlim = [xdata - (xdata - current_xlim[0]) * zoom_factor,
                        xdata + (current_xlim[1] - xdata) * zoom_factor]
            ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event', zoom)
        fig.canvas.mpl_connect('button_press_event', on_click)


class Display:
    def __init__(self, app_instance):
        """
        Initializes the Display class with a reference to the main app instance.

        Parameters:
        - app_instance: The main application instance to access its attributes.
        """
        self.app_instance = app_instance


    def apply_intensity_gradient(self, gray_image):
        settings = SaturationSettings(
            primary_direction=self.app_instance.primary_direction,
            brightness_curve=self.app_instance.brightness_curve,
            brightness=self.app_instance.brightness,
            secondary_direction=self.app_instance.secondary_direction,
            brightness_concave_power=self.app_instance.brightness_concave_power,
            secondary_BrightGain=self.app_instance.secondary_BrightGain,
            saturation_ununiform=self.app_instance.saturation_ununiform,
        )

        return apply_intensity_gradient_gray(gray_image, settings)

    def is_fake_grayscale(self, image, tolerance=5, threshold_percent=99):
        """Returns True if R, G, B channels are nearly equal in most pixels."""
        if image.ndim != 3 or image.shape[2] != 3:
            return False  # Not a 3-channel image

        r = image[..., 0].astype(np.int16)
        g = image[..., 1].astype(np.int16)
        b = image[..., 2].astype(np.int16)

        diff_rg = np.abs(r - g)
        diff_gb = np.abs(g - b)

        mask = (diff_rg < tolerance) & (diff_gb < tolerance)
        percent_equal = 100 * np.sum(mask) / mask.size

        return percent_equal >= threshold_percent



    def display_sub_region(self, sub_region, ROI, Detect_pupil=False):
        # === Remove old items ===
        old = self.app_instance.pupil_ellipse_items
        if old is not None:
            # if it's a tuple/list, remove each; otherwise remove the single item
            if isinstance(old, (tuple, list)):
                for itm in old:
                    self.app_instance.scene2.removeItem(itm)
            else:
                self.app_instance.scene2.removeItem(old)
            self.app_instance.pupil_ellipse_items = None

        # clear any existing pixmap items
        for item in list(self.app_instance.scene2.items()):
            if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                self.app_instance.scene2.removeItem(item)

        # === Build saturation settings once ===
        saturation_settings = SaturationSettings(
            primary_direction=self.app_instance.primary_direction,
            brightness_curve=self.app_instance.brightness_curve,
            brightness=self.app_instance.brightness,
            secondary_direction=self.app_instance.secondary_direction,
            brightness_concave_power=self.app_instance.brightness_concave_power,
            secondary_BrightGain=self.app_instance.secondary_BrightGain,
            saturation_ununiform = self.app_instance.saturation_ununiform,
        )


        # === Apply saturation ===
        if self.app_instance.saturation_method == "Gradual":
            if len(sub_region.shape) == 2 or sub_region.shape[2] == 1:
                # Grayscale image
                processed = apply_intensity_gradient_gray(sub_region, saturation_settings)
            else:
                if self.is_fake_grayscale(sub_region):
                    # Convert fake grayscale to real grayscale
                    gray = cv2.cvtColor(sub_region, cv2.COLOR_BGR2GRAY)
                    processed = apply_intensity_gradient_gray(gray, saturation_settings)
                else:
                    # Real color image
                    processed = change_Gradual_saturation(sub_region, saturation_settings)
                    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)


        elif self.app_instance.saturation_method == "Uniform":
            if sub_region is not None:
                processed = change_saturation_uniform(sub_region, saturation=self.app_instance.saturation,
                                                       contrast=self.app_instance.contrast)
            else:
                raise ValueError("sub_region is None. Cannot apply uniform saturation.")
        else:
            processed = sub_region.copy()

        # === Apply binarization ===
        if self.app_instance.Show_binary:
            if self.app_instance.binary_method == "Adaptive":
                binary = pupil_detection.Image_binarization(processed,self.app_instance.reflect_brightness,self.app_instance.erased_pixels, self.app_instance.reflect_ellipse)
            elif self.app_instance.binary_method == "Constant":
                binary = pupil_detection.Image_binarization_constant(processed,self.app_instance.erased_pixels, self.app_instance.binary_threshold )

            if binary.ndim == 2:  # Grayscale
                sub_region_to_present = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGBA)
            elif binary.ndim == 3 and binary.shape[2] == 3:
                sub_region_to_present = cv2.cvtColor(binary, cv2.COLOR_BGR2RGBA)
            else:
                raise ValueError("Unsupported number of channels in binary image")
        else:

            if processed.ndim == 3 and processed.shape[2] == 3:
                processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                processed_gray = processed

            sub_region_to_present = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGBA)


        # === Display in QGraphicsView ===
        height, width = sub_region_to_present.shape[:2]
        bytes_per_line = width * 4
        qimage = QtGui.QImage(sub_region_to_present.data.tobytes(), width, height, bytes_per_line,
                              QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
        if ROI == "pupil":
            item.setZValue(-1)
        self.app_instance.scene2.addItem(item)
        self.app_instance.scene2.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

        # === Pupil Detection ===
        if Detect_pupil:
            _, center, w, h, angle, _ = pupil_detection.detect_pupil(
                processed,
                self.app_instance.erased_pixels,
                self.app_instance.reflect_ellipse,
                self.app_instance.mnd,
                self.app_instance.reflect_brightness,
                self.app_instance.clustering_method,
                self.app_instance.binary_method,
                self.app_instance.binary_threshold
            )
            ellipse_item = QtWidgets.QGraphicsEllipseItem(
                int(center[0] - w), int(center[1] - h), w * 2, h * 2
            )
            ellipse_item.setTransformOriginPoint(int(center[0]), int(center[1]))
            ellipse_item.setRotation(np.degrees(angle))
            ellipse_item.setPen(QtGui.QPen(QtGui.QColor("purple"), 1))
            self.app_instance.scene2.addItem(ellipse_item)
            r = 3  # radius in pixels
            cx, cy = int(center[0]), int(center[1])
            center_circle = QtWidgets.QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
            center_circle.setPen(QtGui.QPen(QtGui.QColor("yellow"), 1))
            center_circle.setBrush(QtGui.QBrush(QtGui.QColor("yellow")))
            self.app_instance.scene2.addItem(center_circle)

            # keep references if you need to remove later
            self.app_instance.pupil_ellipse_items = (ellipse_item, center_circle)

            ###################
            # === Show as a popped-up OpenCV image ===
            show_img = processed.copy()
            # if len(show_img.shape) == 2:  # grayscale
            #     show_img = cv2.cvtColor(show_img, cv2.COLOR_GRAY2BGR)

            # # Show in popup
            # plt.figure(figsize=(10, 8))
            # if len(show_img.shape) == 3 and show_img.shape[2] == 3:
            #     imgGray = cv2.cvtColor(show_img, cv2.COLOR_BGR2GRAY)
            # else:
            #     imgGray = show_img.copy()
            #
            # plt.imshow(imgGray, cmap='gray')
            # plt.show()

            # Draw the ellipse on a copy
            ellipse_center = (int(center[0]), int(center[1]))
            ellipse_axes = (int(w), int(h))
            ellipse_angle = np.degrees(angle)

            cv2.ellipse(show_img, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, (255, 0, 255), 2)  # purple
            cv2.circle(show_img, ellipse_center, 3, (0, 255, 255), -1)  # yellow center
            if len(show_img.shape) == 3 and show_img.shape[2] == 3:
                show_imgGray = cv2.cvtColor(show_img, cv2.COLOR_BGR2GRAY)
            else:
                show_imgGray = show_img.copy()

            # # # Show in popup
            # plt.figure(figsize=(10, 8))
            # plt.imshow(show_imgGray, cmap='gray')
            # plt.show()
            #

            ###############################

        # === Final render update ===
        if self.app_instance.graphicsView_subImage:
            self.app_instance.graphicsView_subImage.setScene(self.app_instance.scene2)
            self.app_instance.graphicsView_subImage.setFixedSize(scaled_pixmap.width(), scaled_pixmap.height())


        return self.app_instance.pupil_ellipse_items

    def update_frame_view(self, frame):
        """
        Updates the displayed frame in the graphics view and
        handles the display of the pupil and face regions of interest (ROIs).

        Parameters:
        - frame (int): The frame index to be displayed.
        """

        self.app_instance.frame = frame
        if self.app_instance.NPY:
            self.app_instance.image = functions.load_npy_by_index(self.app_instance.folder_path, frame)
        elif self.app_instance.video:
            self.app_instance.image = functions.load_frame_by_index(self.app_instance.cap, frame)

        # Update the displayed frame number
        self.app_instance.lineEdit_frame_number.setText(str(self.app_instance.Slider_frame.value()))

        # Display the main image and update the scene
        self.app_instance.graphicsView_MainFig, self.app_instance.scene = functions.display_region(
            self.app_instance.image, self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width, self.app_instance.image_height, self.app_instance.scene
        )

        # Check if a pupil ROI exists and update its display if present
        if self.app_instance.current_ROI == "pupil":
            self._display_pupil_roi()
        elif self.app_instance.current_ROI == "face":
            self._display_face_roi()
        #################################################
        if hasattr(self.app_instance, 'final_pupil_area') and self.app_instance.final_pupil_area is not None:
            plot_handler = PlotHandler(self.app_instance)
            plot_handler.plot_result(
                self.app_instance.final_pupil_area,
                self.app_instance.graphicsView_pupil,
                label="pupil",
                color="palegreen",
                saccade=self.app_instance.X_saccade,
                Cursor=True
            )

        if hasattr(self.app_instance, 'motion_energy') and self.app_instance.motion_energy is not None:
            plot_handler = PlotHandler(self.app_instance)
            plot_handler.plot_result(
                self.app_instance.motion_energy,
                self.app_instance.graphicsView_whisker,
                label="motion",
                Cursor=True
            )

    def handle_pupil_detection_result(self, result):
        if self.app_instance.pupil_ellipse_items is not None:
            self.app_instance.scene2.removeItem(self.app_instance.pupil_ellipse_items)
            self.app_instance.pupil_ellipse_items = None

        _, P_detected_center, P_detected_width, P_detected_height, angle, _ = result
        ellipse_item = QtWidgets.QGraphicsEllipseItem(
            int(P_detected_center[0] - P_detected_width),
            int(P_detected_center[1] - P_detected_height),
            P_detected_width * 2,
            P_detected_height * 2
        )
        ellipse_item.setTransformOriginPoint(int(P_detected_center[0]), int(P_detected_center[1]))
        ellipse_item.setRotation(np.degrees(angle))
        ellipse_item.setPen(QtGui.QPen(QtGui.QColor("purple"), 1))
        self.app_instance.scene2.addItem(ellipse_item)

        r = 24  # radius in pixels
        cx, cy = int(P_detected_center[0]), int(P_detected_center[1])
        center_circle = QtWidgets.QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
        center_circle.setPen(QtGui.QPen(QtGui.QColor("yellow"), 1))
        center_circle.setBrush(QtGui.QBrush(QtGui.QColor("yellow")))
        self.app_instance.scene2.addItem(center_circle)

        # keep references if you need to remove later
        self.app_instance.pupil_ellipse_items = (ellipse_item, center_circle)

        self.app_instance.pupil_ellipse_items = ellipse_item

    def handle_pupil_error(self, error_msg):
        print(f"[ERROR] Pupil detection failed: {error_msg}")



    def _display_pupil_roi(self):
        """
        Displays the pupil ROI on the graphics view.
        """
        self.app_instance.pupil_ROI = self.app_instance.graphicsView_MainFig.pupil_ROI
        self.app_instance.sub_region, self.app_instance.Pupil_frame = functions.show_ROI(
            self.app_instance.pupil_ROI, self.app_instance.image
        )

        self.display_sub_region( self.app_instance.sub_region,"pupil", Detect_pupil=True)

    def _display_face_roi(self):
        """
        Displays the face ROI on the graphics view.
        """
        self.app_instance.face_ROI = self.app_instance.graphicsView_MainFig.face_ROI
        self.app_instance.sub_region, _ = functions.show_ROI(
            self.app_instance.face_ROI, self.app_instance.image, "face"
        )
        self.display_sub_region(self.app_instance.sub_region,"face", Detect_pupil=False
        )
