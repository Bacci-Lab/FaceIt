import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QSizePolicy

class PlotHandler:
    def __init__(self, app_instance):
        """
        Initializes the PlotHandler with a reference to the main application instance.

        Parameters:
        - app_instance: The main application instance that uses this handler.
        """
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

        self._clear_graphics_view(graphics_view)

        fig, ax = plt.subplots()
        self._plot_main_data(ax, data, label, color)
        self._configure_plot_appearance(ax, fig, data, saccade, background_color, grid, legend_fontsize)
        self._integrate_canvas_into_view(graphics_view, fig)

    def _plot_main_data(self, ax: plt.Axes, data: np.ndarray, label: str, color: str):
        """Plots the main data on the provided axes."""
        x_values = np.arange(len(data))
        ax.plot(x_values, data, color=color, label=label, linestyle='--')

    def _configure_plot_appearance(
        self, ax: plt.Axes, fig: plt.Figure, data: np.ndarray, saccade: np.ndarray,
        background_color: str, grid: bool, legend_fontsize: int
    ):
        """Configures the appearance of the plot."""
        data_min, data_max = np.min(data), np.max(data)
        range_val = data_max - data_min

        if saccade is not None:
            self._plot_saccade(ax, saccade, data_max, range_val)

        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.set_xlim(0, len(data))
        ax.set_ylim(data_min, data_max + range_val / 4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)
        legend = ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)
        for text in legend.get_texts():
            text.set_color("white")

        ax.grid(grid)
        self._setup_interaction_events(fig, ax)

    def _plot_saccade(self, ax: plt.Axes, saccade: np.ndarray, data_max: float, range_val: float):
        """Plots the saccade data as a colormap on the provided axes."""
        y_min = data_max + range_val / 10
        y_max = data_max + range_val / 5
        x_values = np.arange(len(saccade))
        ax.pcolormesh(x_values, [y_min, y_max], saccade, cmap='RdYlGn', shading='flat')

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

    def _setup_interaction_events(self, fig: plt.Figure, ax: plt.Axes):
        """Sets up zoom, pan, and click events for the plot."""
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
            ax.set_xlim(ax.get_xlim()[0] - dx, ax.get_xlim()[1] - dx)
            fig.canvas.draw_idle()

        def on_release(event):
            self.panning = False
            self.press_event = None
            event.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        def on_click(event):
            """Handles click events for setting grooming threshold."""
            if getattr(self.app_instance, 'find_grooming_threshold', False):
                if event.inaxes == ax and event.ydata is not None:
                    self.app_instance.grooming_thr = event.ydata
                    print(f"Clicked at y: {event.ydata:.2f}")
                    self.app_instance.find_grooming_threshold = False
                    if hasattr(self.app_instance, 'lineEdit_grooming_y'):
                        self.app_instance.lineEdit_grooming_y.setText(str(int(event.ydata)))
                    if hasattr(self.app_instance, 'display_removed_grooming'):
                        self.app_instance.display_removed_grooming(self.app_instance.grooming_thr, getattr(self.app_instance, 'motion_energy', []))

        def zoom(event):
            """Handles zooming based on mouse scroll events."""
            current_xlim = ax.get_xlim()
            if event.xdata is None:
                return
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            new_xlim = [event.xdata - (event.xdata - current_xlim[0]) * zoom_factor,
                        event.xdata + (current_xlim[1] - event.xdata) * zoom_factor]
            ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event', zoom)
        fig.canvas.mpl_connect('button_press_event', on_click)
