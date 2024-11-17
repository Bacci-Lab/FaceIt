from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create a central widget and set a layout
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Create the button
        self.brush_button = QtWidgets.QPushButton("Activate Brush")
        layout.addWidget(self.brush_button)

        # Create the graphics view and scene
        self.graphics_view = CustomGraphicsView()
        layout.addWidget(self.graphics_view)

        # Set layout and central widget
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect the button to the function
        self.brush_button.clicked.connect(self.graphics_view.toggleBrushMode)

        # Setup window properties
        self.setWindowTitle("Brush Tool Example")
        self.resize(800, 600)

class CustomGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene2 = QtWidgets.QGraphicsScene()
        self.setScene(self.scene2)
        self.brush_active = False
        self.brush_color = QtGui.QColor('black')
        self.erase_size = 10  # Brush size

    def toggleBrushMode(self):
        """Toggle the brush mode on or off."""
        self.brush_active = not self.brush_active
        if self.brush_active:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def mousePressEvent(self, event):
        """Handle mouse press event to start painting if the brush mode is active."""
        if self.brush_active and event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.paint(scene_pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move event for continuous painting."""
        if self.brush_active and event.buttons() & QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.paint(scene_pos)
        super().mouseMoveEvent(event)

    def paint(self, scene_pos):
        """Paint a circle at the given scene position."""
        brush_item = QtWidgets.QGraphicsEllipseItem(
            scene_pos.x() - self.erase_size / 2,
            scene_pos.y() - self.erase_size / 2,
            self.erase_size,
            self.erase_size
        )
        brush_item.setBrush(QtGui.QBrush(self.brush_color))
        brush_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scene2.addItem(brush_item)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
