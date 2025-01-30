import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QSizePolicy
from FACEIT_codes import functions
class PlotHandler:
    def __init__(self,app_instance):
        self.app_instance = app_instance
        self.panning = False
        self.press_event = None

    def plot_result(
            self,
            data: np.ndarray,
            graphics_view: QtWidgets.QGraphicsView,
            label: str,
            color: str = '#D97A53',
            saccade: np.ndarray = None,
            background_color: str = '#3d4242',
            grid: bool = False,
            legend_fontsize: int = 8
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

        # Clear the graphics view and set up the canvas
        self._clear_graphics_view(graphics_view)

        # Create figure and axes
        fig, ax = plt.subplots()
        self._plot_data(ax, data, label, color)
        self._plot_saccade(ax, saccade, data)

        # Customize the plot
        self._customize_plot(fig, ax, data, background_color, grid, legend_fontsize)

        # Integrate the plot into the graphics view
        self._integrate_canvas_into_view(graphics_view, fig)

    def _plot_data(self, ax: plt.Axes, data: np.ndarray, label: str, color: str):
        """Plots the main data on the provided axes."""
        x_values = np.arange(len(data))
        ax.plot(x_values, data, color=color, label=label, linestyle='--')

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
        ax.set_xlim(0, len(data))
        ax.set_ylim(data_min, data_max + range_val / 4)

        # Customize axes and ticks
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)

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
        layout.addWidget(canvas)
        graphics_view.setLayout(layout)

        fig.tight_layout(pad=0)
        fig.subplots_adjust(bottom=0.15)
        canvas.draw()

    def _integrate_canvas_into_view(self, graphics_view: QtWidgets.QGraphicsView, fig: plt.Figure):
        """Integrates the Matplotlib canvas into the PyQt graphics view."""
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()

        layout = QtWidgets.QVBoxLayout(graphics_view)
        layout.addWidget(canvas)
        graphics_view.setLayout(layout)

        fig.tight_layout(pad=0)
        fig.subplots_adjust(bottom=0.15)
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

    def update_frame_view(self, frame):
        """
        Updates the displayed frame in the graphics view and handles the display of the pupil and face regions of interest (ROIs).

        Parameters:
        - frame (int): The frame index to be displayed.
        """
        self.app_instance.frame = frame

        # Load the image based on the data type (NPY or video)
        if self.app_instance.NPY:
            self.app_instance.image = functions.load_npy_by_index(self.app_instance.folder_path, frame)
        elif self.app_instance.video:
            self.app_instance.image = functions.load_frame_by_index(self.app_instance.folder_path, frame)

        # Update the displayed frame number
        self.app_instance.lineEdit_frame_number.setText(str(self.app_instance.Slider_frame.value()))

        # Display the main image and update the scene
        self.app_instance.graphicsView_MainFig, self.app_instance.scene = functions.display_region(
            self.app_instance.image, self.app_instance.graphicsView_MainFig,
            self.app_instance.image_width, self.app_instance.image_height, self.app_instance.scene
        )

        # Check if a pupil ROI exists and update its display if present
        if self.app_instance.Pupil_ROI_exist:
            self._display_pupil_roi()
        # Check if a face ROI exists and update its display if present
        elif self.app_instance.Face_ROI_exist:
            self._display_face_roi()

    def _display_pupil_roi(self):
        """
        Displays the pupil ROI on the graphics view.
        """
        self.app_instance.pupil_ROI = self.app_instance.graphicsView_MainFig.pupil_ROI
        self.app_instance.sub_region, self.app_instance.Pupil_frame = functions.show_ROI(
            self.app_instance.pupil_ROI, self.app_instance.image
        )
        self.app_instance.pupil_ellipse_items = functions.display_sub_region(
            self.app_instance.graphicsView_subImage, self.app_instance.sub_region,
            self.app_instance.scene2, "pupil area", self.app_instance.saturation,
            self.app_instance.erased_pixels, self.app_instance.reflect_ellipse,
            self.app_instance.pupil_ellipse_items, Detect_pupil=True
        )

    def _display_face_roi(self):
        """
        Displays the face ROI on the graphics view.
        """
        self.app_instance.face_ROI = self.app_instance.graphicsView_MainFig.face_ROI
        self.app_instance.sub_region, _ = functions.show_ROI(
            self.app_instance.face_ROI, self.app_instance.image
        )
        functions.display_sub_region(
            self.app_instance.graphicsView_subImage, self.app_instance.sub_region,
            self.app_instance.scene2, "face", self.app_instance.saturation,
            self.app_instance.erased_pixels, self.app_instance.reflect_ellipse,
            self.app_instance.pupil_ellipse_items, Detect_pupil=False
        )
