from PyQt5 import QtWidgets, QtCore, QtGui
import functions
class GUI_Intract(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.setMouseTracking(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.sub_region = None

        self.dragging = False
        self.dragging_face = False
        self.dragging_pupil = False
        self.dragging_reflect = False

        self.eye_corner_mode = False
        self.Resizing = False
        self.Resize_face = False
        self.Resize_pupil = False
        self.face_ROI = None
        self.eyecorner = None
        self.pupil_ROI = None
        self.eye_corner_center = None
        self.oval_width = 100
        self.rect_width = 100
        self.ROI_width = 100
        self.oval_height = 50
        self.ROI_height = 50
        self.rect_height = 50
        self.offset = QtCore.QPoint()
        self.scene_pos = None
        self.previous_mouse_pos = None
        self.Face_frame = None
        self.Pupil_frame = None
        self.current_ROI = None
        #----------------------------------- initiate reflection-----------------------------------
        self.Resize_reflect = False
        self.reflect_ROI = None
        self.reflect_ROIs = []
        self.reflect_handles_list = []
        self.reflect_widths = []
        self.reflect_heights = []
        self.reflect_centers = []
        self.reflect_ellipse = None
        self.pupil_ellipse_items = None

        self.erase_size = 5
        self.brush_strokes = []
        self.erased_pixels = []
        #####################
        self.added_pixels = []
        self.reflect_strokes = []

    def showContextMenu(self, pos):
        context_menu = QtWidgets.QMenu(self)
        delete_action = context_menu.addAction("Delete ROI")
        action = context_menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.scene_pos = self.mapToScene(pos)
            self.delete(self.scene_pos)



    def delete(self, scene_pos):

        for idx, reflect_ROI in enumerate(self.reflect_ROIs):
            reflect_handle = self.reflect_handles_list[idx]
            if reflect_ROI.contains(scene_pos):
                self.scene().removeItem(reflect_ROI)
                self.scene().removeItem(reflect_handle['right'])
                del self.reflect_ROIs[idx]
                del self.reflect_handles_list[idx]
                del self.reflect_centers[idx]
                del self.reflect_widths[idx]
                del self.reflect_heights[idx]
                break

    def _handle_eye_corner_mode(self):
        """Handle eye corner mode for adding an eye corner."""
        self.parent.eye_corner_center = functions.add_eyecorner(
            self.scene_pos.x(), self.scene_pos.y(),
            self.parent.scene2, self.parent.graphicsView_subImage
        )
        self.parent.eye_corner_mode = False

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.parent.Eraser_active and event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.markForErasure(scene_pos)
        elif self.parent.AddPixels_active and event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.markForAddition(scene_pos)

        self.scene_pos = self.mapToScene(event.pos())
        if event.button() == QtCore.Qt.RightButton:
            return
        if self.parent and hasattr(self.parent, 'eye_corner_mode') and self.parent.eye_corner_mode:
            self._handle_eye_corner_mode()
            return


        if self.pupil_ROI:
            for handle_name, handle in self.pupil_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_pupil = True

                    self.previous_mouse_pos_pupil = (event.pos().x(), event.pos().y())
                    return

            if self.pupil_ROI.contains(self.scene_pos):
                self.parent.current_ROI = "pupi"
                self.dragging = True
                self.dragging_pupil = True
                self.previous_mouse_pos_pupil = (event.pos().x(), event.pos().y())

                return

        if self.face_ROI:
            for handle_name, handle in self.face_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_face = True

                    self.previous_mouse_pos_face = (event.pos().x(), event.pos().y())
                    return

            if self.face_ROI.contains(self.scene_pos):
                self.parent.current_ROI = "face"
                self.dragging = True
                self.dragging_face = True
                self.previous_mouse_pos_face = (event.pos().x(), event.pos().y())

                return

        for idx, self.reflect_ROI in enumerate(self.reflect_ROIs):
            reflect_handles = self.reflect_handles_list[idx]
            for handle_name, handle in reflect_handles.items():
                if handle.contains(self.scene_pos):
                    self.Resizing = True
                    self.Resize_reflect = True

                    self.current_reflect_idx = idx
                    self.previous_mouse_pos_reflect = (self.scene_pos .x(), self.scene_pos .y())
                    return

            if self.reflect_ROI.contains(self.scene_pos):
                self.dragging = True
                self.dragging_reflect = True

                self.current_reflect_idx = idx
                self.previous_mouse_pos_reflect = (self.scene_pos .x(), self.scene_pos .y())



        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        print("self.parent.Eraser_active", self.parent.Eraser_active)
        print("self.parent.AddPixels_active", self.parent.AddPixels_active)
        """Handle mouse move events for continuous painting."""
        if self.parent.Eraser_active and event.buttons() & QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.markForErasure(scene_pos)

        ######################
        if self.parent.AddPixels_active and event.buttons() & QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.markForAddition(scene_pos)
        #######################################

        if self.dragging:
            if self.dragging_face:
                handle_type = "face"
                previous_mouse_pos = self.previous_mouse_pos_face
                self.ROI_center = self.parent.face_rect_center
                self.ROI_width = self.rect_width
                self.ROI_height = self.rect_height
                frame_height_boundary =  self.parent.image_height
                frame_width_boundary = self.parent.image_width

            elif self.dragging_pupil:
                handle_type = "pupil"
                previous_mouse_pos = self.previous_mouse_pos_pupil
                self.ROI_center = self.parent.oval_center
                self.ROI_width = self.oval_width
                self.ROI_height = self.oval_height
                frame_height_boundary = self.parent.image_height
                frame_width_boundary = self.parent.image_width

            elif self.dragging_reflect:
                handle_type = 'reflection'
                previous_mouse_pos = self.previous_mouse_pos_reflect
                self.ROI_center = self.reflect_centers[self.current_reflect_idx]
                self.ROI_height = self.reflect_heights[self.current_reflect_idx]
                self.ROI_width = self.reflect_widths[self.current_reflect_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]


            x = previous_mouse_pos[0]
            y = previous_mouse_pos[1]

            new_pos = self.mapToScene(event.pos())
            self.x_offset = new_pos.x() - x
            self.y_offset = new_pos.y() - y
            # Boundary checks
            half_width = self.ROI_width / 2
            half_height = self.ROI_height / 2
            center_x = self.ROI_center[0] + self.x_offset
            center_y = self.ROI_center[1] + self.y_offset

            if center_x >= frame_width_boundary - half_width:
                center_x = frame_width_boundary - half_width
            elif center_x <= half_width:
                center_x = half_width
            if center_y >= frame_height_boundary - half_height:
                center_y = frame_height_boundary - half_height
            elif center_y <= half_height:
                center_y = half_height

            self.updateEllipse(center_x, center_y, self.ROI_width, self.ROI_height, handle_type)
            if self.dragging_face:
                self.previous_mouse_pos_face = (new_pos.x(), new_pos.y())
                self.parent.face_rect_center = (center_x, center_y)
            elif self.dragging_pupil:
                self.previous_mouse_pos_pupil = (new_pos.x(), new_pos.y())
                self.parent.oval_center = (center_x, center_y)
            elif self.dragging_reflect:
                self.previous_mouse_pos_reflect = (new_pos.x(), new_pos.y())
                self.reflect_centers[self.current_reflect_idx] = (center_x, center_y)


        elif self.Resizing:
            if self.Resize_face:
                handle_type = "face"
                previous_mouse_pos = self.previous_mouse_pos_face
                self.ROI_center = self.parent.face_rect_center
                self.ROI_width = self.rect_width
                self.ROI_height = self.rect_height
                frame_width_boundary = self.parent.image_width
                minimum_w_h = 10

            elif self.Resize_pupil:
                handle_type = "pupil"
                previous_mouse_pos = self.previous_mouse_pos_pupil
                self.ROI_center = self.parent.oval_center
                self.ROI_width = self.oval_width
                self.ROI_height = self.oval_height
                frame_width_boundary = self.parent.image_width
                minimum_w_h = 10

            elif self.Resize_reflect:
                handle_type = "reflection"
                previous_mouse_pos = self.previous_mouse_pos_reflect
                self.ROI_center = self.reflect_centers[self.current_reflect_idx]
                self.ROI_width = self.reflect_widths[self.current_reflect_idx]
                self.ROI_height = self.reflect_heights[self.current_reflect_idx]
                frame_height_boundary = self.parent.sub_region.shape[0]
                frame_width_boundary = self.parent.sub_region.shape[1]
                minimum_w_h = 1


            x = previous_mouse_pos[0]
            y = previous_mouse_pos[1]
            new_pos = self.mapToScene(event.pos())
            self.x_offset = new_pos.x() - x
            self.y_offset = new_pos.y() - y
            resized_width = self.ROI_width + 2 * (self.x_offset)
            resized_height = self.ROI_height - 2 * (self.y_offset)

            # Ensure minimum size constraints
            if resized_width < minimum_w_h:
                resized_width = minimum_w_h
            if resized_height < minimum_w_h:
                resized_height = minimum_w_h
            if self.ROI_center[0] + resized_width / 2 >= frame_width_boundary:
                resized_width = (frame_width_boundary - self.ROI_center[0]) * 2
            if self.ROI_center[1] <= resized_height / 2:
                resized_height = self.ROI_center[1] * 2

            self.updateEllipse(self.ROI_center[0], self.ROI_center[1], resized_width, resized_height, handle_type)

            if self.Resize_face:
                self.previous_mouse_pos_face = (new_pos.x(), new_pos.y())
                self.rect_width = resized_width
                self.rect_height = resized_height

            elif self.Resize_pupil:
                self.previous_mouse_pos_pupil = (new_pos.x(), new_pos.y())
                self.oval_width = resized_width
                self.oval_height = resized_height

            elif self.Resize_reflect:
                self.previous_mouse_pos_reflect = (new_pos.x(), new_pos.y())
                self.reflect_heights[self.current_reflect_idx] = resized_height
                self.reflect_widths[self.current_reflect_idx] = resized_width

        super().mouseMoveEvent(event)



    def mouseReleaseEvent(self, event):
        if self.dragging:
            if self.dragging_face:
                self.sub_region, self.parent.Face_frame = functions.show_ROI(self.face_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.parent.scene2, "face",self.parent.saturation)
                self.parent.set_frame(self.parent.Face_frame)
            elif self.dragging_pupil:
                self.parent.sub_region, self.parent.Pupil_frame = functions.show_ROI(self.pupil_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.parent.sub_region, self.parent.scene2,"pupil",self.parent.saturation)
                self.parent.set_frame(self.parent.Pupil_frame)
                self.parent.reflection_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)
            elif self.dragging_reflect:
                self.reflect_ellipse = [self.reflect_centers, self.reflect_widths , self.reflect_heights ]
                self.parent.reflect_ellipse = self.reflect_ellipse

        elif self.Resizing:
            if self.Resize_face:
                self.sub_region, self.parent.Face_frame = functions.show_ROI(self.face_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.sub_region, self.parent.scene2, "face",self.parent.saturation)
                self.parent.set_frame(self.parent.Face_frame)
            elif self.Resize_pupil:
                self.parent.sub_region, self.parent.Pupil_frame = functions.show_ROI(self.pupil_ROI, self.parent.image)
                _ = functions.display_sub_region(self.graphicsView_subImage, self.parent.sub_region, self.parent.scene2, "pupil",self.parent.saturation)
                self.parent.set_frame(self.parent.Pupil_frame)
                self.parent.reflection_center = (
                    (self.parent.Pupil_frame[3] - self.parent.Pupil_frame[2]) / 2, (self.parent.Pupil_frame[1] - self.parent.Pupil_frame[0]) / 2)
            elif self.Resize_reflect:
                self.reflect_ellipse = [self.reflect_centers, self.reflect_widths , self.reflect_heights]
                self.parent.reflect_ellipse = self.reflect_ellipse
        if self.dragging or self.Resizing:
            self._reset_flags()

        super().mouseReleaseEvent(event)

    def _reset_flags(self):
        """Reset all dragging and resizing flags."""
        self.dragging = False
        self.dragging_face = False
        self.dragging_pupil = False
        self.dragging_reflect = False
        self.Resizing = False
        self.Resize_face = False
        self.Resize_pupil = False
        self.Resize_reflect = False

    def updateEllipse(self, center_x, center_y, width, height, handle_type):
        """
        Update the position and size of an ROI (Region of Interest) and its handles based on the given type.

        Args:
            center_x (float): The x-coordinate of the ellipse center.
            center_y (float): The y-coordinate of the ellipse center.
            width (float): The width of the ellipse.
            height (float): The height of the ellipse.
            handle_type (str): The type of ROI being updated ('face', 'pupil', 'reflection').
        """
        # Calculate the rectangle's top-left corner for the given center and size.
        rect_x = center_x - width / 2
        rect_y = center_y - height / 2

        # Update the ROI and its handles based on the type.
        if handle_type == "face":
            self._update_ROI(self.face_ROI, rect_x, rect_y, width, height, "face", handle_size=10)
        elif handle_type == "pupil":
            self._update_ROI(self.pupil_ROI, rect_x, rect_y, width, height, "pupil", handle_size=10)
        elif handle_type == "reflection":
            current_reflect_ROI = self.reflect_ROIs[self.current_reflect_idx]
            self._update_ROI(current_reflect_ROI, rect_x, rect_y, width, height, "reflection", handle_size=3)
            current_reflect_ROI.setBrush(QtGui.QBrush(QtGui.QColor('silver')))

    def _update_ROI(self, roi, rect_x, rect_y, width, height, handle_type, handle_size):
        """
        Update the given ROI and its associated handles.

        Args:
            roi (QGraphicsRectItem): The ROI to update.
            rect_x (float): The x-coordinate of the rectangle's top-left corner.
            rect_y (float): The y-coordinate of the rectangle's top-left corner.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            handle_type (str): The type of ROI being updated.
            handle_size (int): The size of the handles.
        """
        if roi:
            roi.setRect(rect_x, rect_y, width, height)
            self.updateHandles(rect_x + width / 2, rect_y + height / 2, handle_type, handle_size)
    def updateHandles(self, center_x, center_y, handle_type, handle_size):
        """
        Update the position of handles for the specified ROI type.

        Args:
            center_x (float): The x-coordinate of the center of the ROI.
            center_y (float): The y-coordinate of the center of the ROI.
            handle_type (str): The type of ROI ('face', 'pupil', 'reflection').
            handle_size (int): The size of the handle rectangle.
        """
        # Calculate half the width of the ROI for positioning handles.
        half_width = self.ROI_width / 2

        # Select the appropriate handles based on the handle type.
        handles = self._get_handles(handle_type)

        # Update the 'right' handle's position.
        if 'right' in handles:
            handles['right'].setRect(
                center_x + half_width - handle_size // 2,  # x-coordinate of the handle's top-left corner
                center_y - handle_size // 2,  # y-coordinate of the handle's top-left corner
                handle_size,  # width of the handle
                handle_size  # height of the handle
            )

    def _get_handles(self, handle_type):
        """
        Retrieve the appropriate handle dictionary based on the handle type.

        Args:
            handle_type (str): The type of ROI ('face', 'pupil', 'reflection').

        Returns:
            dict: A dictionary of handles for the specified ROI type.
        """
        if handle_type == "face":
            return self.face_handles
        elif handle_type == "pupil":
            return self.pupil_handles
        elif handle_type == "reflection":
            return self.reflect_handles_list[self.current_reflect_idx]
        else:
            raise ValueError(f"Invalid handle type: {handle_type}")

    def resetModes(self):
        """Deactivate all modes."""
        self.parent.Eraser_active = False
        self.parent.AddPixels_active = False
        # Add other mode flags here as needed

    def activateEraseMode(self):
        self.resetModes()  # Reset all modes first
        self.parent.Eraser_active = True  # Activate Erase mode
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    # Method to activate Add Pixels mode
    def activateAddPixelsMode(self):
        self.resetModes()  # Reset all modes first
        self.parent.AddPixels_active = True  # Activate Add Pixels mode
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))


    def markForErasure(self, scene_pos):
        """Paint a circle at the given scene position."""
        brush_item = QtWidgets.QGraphicsEllipseItem(
            scene_pos.x() - self.erase_size / 2,
            scene_pos.y() - self.erase_size / 2,
            self.erase_size,
            self.erase_size
        )
        brush_item.setBrush(QtGui.QBrush(QtGui.QColor('white')))
        brush_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.parent.scene2.addItem(brush_item)
        #########

        # Store the coordinates of the painted pixels
        for x in range(int(scene_pos.x() - self.erase_size / 2), int(scene_pos.x() + self.erase_size / 2)):
            for y in range(int(scene_pos.y() - self.erase_size / 2), int(scene_pos.y() + self.erase_size / 2)):
                # Only store coordinates within the bounds of the scene
                if 0 <= x < self.parent.scene2.width() and 0 <= y < self.parent.scene2.height():
                    self.erased_pixels.append((x, y))
                    self.parent.erased_pixels = self.erased_pixels
        self.brush_strokes.append(brush_item)

    def undoBrushStrokes(self):
        """Remove all brush strokes from the scene."""
        while self.brush_strokes:
            brush_item = self.brush_strokes.pop()
            self.parent.scene2.removeItem(brush_item)
            self.parent.erased_pixels = None


    #########################
    def markForAddition(self, scene_pos):
        """Paint a circle at the given scene position for added pixels."""
        brush_item = QtWidgets.QGraphicsEllipseItem(
            scene_pos.x() - self.erase_size / 2,
            scene_pos.y() - self.erase_size / 2,
            self.erase_size,
            self.erase_size
        )
        brush_item.setBrush(QtGui.QBrush(QtGui.QColor('gray')))
        brush_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.parent.scene2.addItem(brush_item)

        # Store the coordinates of the added pixels
        for x in range(int(scene_pos.x() - self.erase_size / 2), int(scene_pos.x() + self.erase_size / 2)):
            for y in range(int(scene_pos.y() - self.erase_size / 2), int(scene_pos.y() + self.erase_size / 2)):
                if 0 <= x < self.parent.scene2.width() and 0 <= y < self.parent.scene2.height():
                    self.added_pixels.append((x, y))
                    self.parent.added_pixels = self.added_pixels  # Sync with parent

        self.reflect_strokes.append(brush_item)

    def undoAddedPixels(self):
        """Remove all reflect brush  strokes from the scene."""
        while self.reflect_strokes:
            reflect_item = self.reflect_strokes.pop()
            self.parent.scene2.removeItem(reflect_item)
            self.parent.added_pixels = None


