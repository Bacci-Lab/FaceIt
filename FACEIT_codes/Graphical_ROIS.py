from PyQt5 import QtWidgets, QtGui
from FACEIT_codes import functions

class ROIHandler:
    def __init__(self, app_instance):
        """
        Initializes the ROIHandler with scene and graphics view instances.
        """
        self.app_instance = app_instance

    def enable_button(self, button):
        """Enables the specified button."""
        if button:
            button.setEnabled(True)

    def disable_button(self, button):
        """Disables the specified button."""
        if button:
            button.setEnabled(False)

    def Add_ROI(self, roi_type, roi_center, image, **kwargs):
        """
        Adds a Region of Interest (ROI) to the appropriate scene based on the type.

        Parameters:
        - roi_type (str): The type of ROI to add ('pupil', 'face', 'reflection',  'pupil_detection').
        - roi_center (tuple): The center coordinates for the ROI.
        - image (ndarray): The image being processed.
        - kwargs: Additional arguments for customization (e.g., dimensions, buttons, etc.).
        """
        saturation = 0
        contrast = 1
        roi_item, handles = self._draw_roi(roi_center, roi_type, kwargs.get('height', 50), kwargs.get('width', 80), kwargs.get('handle_size', 10), color=kwargs.get('color', 'gold'))

        if roi_type in ['pupil', 'face', 'pupil_detection']:
            self.app_instance.scene.addItem(roi_item)
            for handle in handles.values():
                self.app_instance.scene.addItem(handle)

            if roi_type == 'pupil':
                self.app_instance.graphicsView_MainFig.pupil_handles = handles
                self.app_instance.graphicsView_MainFig.pupil_ROI = roi_item

                self._process_roi_display(roi_item, image, roi_type, saturation,contrast)
                self.enable_button(kwargs.get('Button'))
                self.enable_button(kwargs.get('Button2'))
                self.enable_button(kwargs.get('Button4'))
                self.enable_button(kwargs.get('checkBox_pupil'))
                self.enable_button(kwargs.get('Button5'))
                self.disable_button(kwargs.get('Button3'))

            elif roi_type == 'face':
                self.app_instance.graphicsView_MainFig.face_handles = handles
                self.app_instance.graphicsView_MainFig.face_ROI = roi_item
                self._process_roi_display(roi_item, image, roi_type, saturation, contrast)
                self.enable_button(kwargs.get('Button4'))
                self.disable_button(kwargs.get('Button3'))
                self.enable_button(kwargs.get('checkBox_face'))

            elif roi_type == 'pupil_detection':
                self.app_instance.graphicsView_MainFig.pupil_detection = roi_item
                self._process_roi_display(roi_item, image, roi_type, saturation, contrast)

        elif roi_type == 'reflection':
            self._add_to_scene2(roi_item, handles, 'reflect', roi_center, kwargs)

    def _draw_roi(self, center, roi_type, height, width, handle_size, color='gold'):
        """
        Draws an ROI item based on type.

        Parameters:
        - center (tuple): Center coordinates for the ROI.
        - roi_type (str): The type of ROI ('pupil', 'face', etc.).
        - height (int): Height of the ROI.
        - width (int): Width of the ROI.
        - handle_size (int): Size of the handle.
        - color (str): Border color for the ROI.

        Returns:
        - QtWidgets.QGraphicsItem: The ROI item.
        - dict: A dictionary containing handles for the ROI.
        """
        color2 = 'teal' if roi_type in ['pupil', 'face'] else 'gray'
        ROI = (QtWidgets.QGraphicsEllipseItem if roi_type in ['pupil', 'reflection', 'pupil_detection'] else QtWidgets.QGraphicsRectItem)(
            center[0] - width / 2, center[1] - height / 2, width, height
        )
        pen = QtGui.QPen(QtGui.QColor(color))
        pen.setWidth(0)
        ROI.setPen(pen)

        handles = {
            'right': QtWidgets.QGraphicsRectItem(center[0] + width // 2 - handle_size // 2, center[1] - handle_size // 2, handle_size, handle_size)
        }
        handle_pen = QtGui.QPen(QtGui.QColor(color2))
        handle_pen.setWidth(0)
        for handle in handles.values():
            handle.setPen(handle_pen)

        return ROI, handles

    def _process_roi_display(self, roi, image, roi_type, saturation, contrast):
        """
        Processes and displays a sub-region from the ROI.

        Parameters:
        - roi: The ROI item.
        - image: The image being processed.
        - roi_type: Type of the ROI.
        - saturation: Saturation value for display.
        - kwargs: Additional arguments.
        """
        sub_region, _ = functions.show_ROI(roi, image)
        functions.display_sub_region(
            self.app_instance.graphicsView_subImage,
            sub_region,
            self.app_instance.scene2,
            roi_type,
            saturation,
            contrast,
            self.app_instance.mnd,
            self.app_instance.binary_threshold
        )

    def _add_to_scene2(self, roi, handles, roi_list_attr, center, kwargs):
        """
        Adds an ROI and its handles to `scene2` and updates the graphics view attributes.

        Parameters:
        - roi: The ROI item.
        - handles: Handles associated with the ROI.
        - roi_list_attr: The attribute name in `graphicsView_subImage` to update.
        - center: Center coordinates of the ROI.
        - kwargs: Additional arguments.
        """
        self.app_instance.scene2.addItem(roi)
        for handle in handles.values():
            self.app_instance.scene2.addItem(handle)

        attr_list = getattr(self.app_instance.graphicsView_subImage, f"{roi_list_attr}_ROIs", [])
        attr_handles_list = getattr(self.app_instance.graphicsView_subImage, f"{roi_list_attr}_handles_list", [])
        attr_centers = getattr(self.app_instance.graphicsView_subImage, f"{roi_list_attr}_centers", [])
        attr_heights = getattr(self.app_instance.graphicsView_subImage, f"{roi_list_attr}_heights", [])
        attr_widths = getattr(self.app_instance.graphicsView_subImage, f"{roi_list_attr}_widths", [])
        attr_list.append(roi)
        attr_handles_list.append(handles)
        attr_centers.append(center)
        attr_heights.append(kwargs.get('height'))
        attr_widths.append(kwargs.get('width'))
