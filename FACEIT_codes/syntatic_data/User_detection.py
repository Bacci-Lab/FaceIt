# from PyQt5 import QtCore, QtWidgets, QtGui
# from openpyxl import Workbook
# import os
#
#
# class ExampleWindow(QtWidgets.QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Example Images")
#         self.resize(800, 600)
#
#         layout = QtWidgets.QVBoxLayout(self)
#
#         # Single Section: Mixed Examples
#         label_examples = QtWidgets.QLabel("<b>Good Detection </b>")
#         layout.addWidget(label_examples, alignment=QtCore.Qt.AlignCenter)
#
#         mixed_layout = QtWidgets.QHBoxLayout()
#         layout.addLayout(mixed_layout)
#
#         # Combine all images into one list
#         mixed_images = [
#             # Good ones
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good1.png",
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good2.png",
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good3.png",
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good4.png",
#             # r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\inaccurate_example_3.png",
#         ]
#
#         for img_path in mixed_images:
#             lbl = QtWidgets.QLabel()
#             pixmap = QtGui.QPixmap(img_path).scaled(200, 200, QtCore.Qt.KeepAspectRatio)
#             lbl.setPixmap(pixmap)
#             mixed_layout.addWidget(lbl)
#
#         # Section C: Bad
#         label_bad = QtWidgets.QLabel("<b>Bad Detection</b>")
#         layout.addWidget(label_bad, alignment=QtCore.Qt.AlignCenter)
#         bad_layout = QtWidgets.QHBoxLayout()
#         layout.addLayout(bad_layout)
#         bad_images = [
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad1.png",
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad2.png",
#             r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad3.png",
#         ]
#         for img_path in bad_images:
#             lbl = QtWidgets.QLabel()
#             pixmap = QtGui.QPixmap(img_path).scaled(200, 200, QtCore.Qt.KeepAspectRatio)
#             lbl.setPixmap(pixmap)
#             bad_layout.addWidget(lbl)
#
#
# class FullFrameWindow(QtWidgets.QDialog):
#     def __init__(self, img_path, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Full Frame")
#         self.resize(800, 600)
#
#         layout = QtWidgets.QVBoxLayout(self)
#         lbl = QtWidgets.QLabel()
#         pixmap = QtGui.QPixmap(img_path).scaled(750, 550, QtCore.Qt.KeepAspectRatio)
#         lbl.setPixmap(pixmap)
#         layout.addWidget(lbl, alignment=QtCore.Qt.AlignCenter)
#
#
# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.resize(650, 480)
#         self.ratings = {}
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#
#         # GraphicsView
#         self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
#         self.graphicsView.setGeometry(QtCore.QRect(220, 0, 400, 370))
#
#         # Filename display
#         self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
#         self.lineEdit_filename.setGeometry(QtCore.QRect(220, 380, 400, 20))
#         self.lineEdit_filename.setReadOnly(True)
#
#         # Name fields
#         self.lineEdit_FirstName = QtWidgets.QLineEdit(self.centralwidget)
#         self.lineEdit_FirstName.setGeometry(QtCore.QRect(80, 10, 91, 20))
#         self.lineEdit_LastName = QtWidgets.QLineEdit(self.centralwidget)
#         self.lineEdit_LastName.setGeometry(QtCore.QRect(80, 40, 91, 20))
#
#         # Labels
#         self.label = QtWidgets.QLabel("Name", self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(10, 10, 60, 20))
#         self.label_2 = QtWidgets.QLabel("Last Name", self.centralwidget)
#         self.label_2.setGeometry(QtCore.QRect(10, 40, 60, 20))
#
#         # Buttons
#         self.pushButton_2 = QtWidgets.QPushButton("Good", self.centralwidget)
#         self.pushButton_2.setGeometry(QtCore.QRect(150, 140, 61, 23))
#         self.pushButton_4 = QtWidgets.QPushButton("Bad", self.centralwidget)
#         self.pushButton_4.setGeometry(QtCore.QRect(10, 140, 61, 23))
#         self.pushButton_5 = QtWidgets.QPushButton("Save", self.centralwidget)
#         self.pushButton_5.setGeometry(QtCore.QRect(30, 340, 161, 31))
#         self.pushButton_6 = QtWidgets.QPushButton("Previous", self.centralwidget)
#         self.pushButton_6.setGeometry(QtCore.QRect(20, 220, 75, 23))
#         self.pushButton_7 = QtWidgets.QPushButton("Next", self.centralwidget)
#         self.pushButton_7.setGeometry(QtCore.QRect(120, 220, 75, 23))
#         self.pushButton_fullFrame = QtWidgets.QPushButton("Show Full Frame", self.centralwidget)
#         self.pushButton_fullFrame.setGeometry(QtCore.QRect(65, 275, 120, 30))
#
#         # Horizontal lines
#         self.line = QtWidgets.QFrame(self.centralwidget)
#         self.line.setGeometry(QtCore.QRect(10, 120, 200, 20))
#         self.line.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_2 = QtWidgets.QFrame(self.centralwidget)
#         self.line_2.setGeometry(QtCore.QRect(10, 170, 200, 20))
#         self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_3 = QtWidgets.QFrame(self.centralwidget)
#         self.line_3.setGeometry(QtCore.QRect(10, 240, 200, 20))
#         self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_4 = QtWidgets.QFrame(self.centralwidget)
#         self.line_4.setGeometry(QtCore.QRect(10, 200, 200, 20))
#         self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_5 = QtWidgets.QFrame(self.centralwidget)
#         self.line_5.setGeometry(QtCore.QRect(10, 320, 200, 20))
#         self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
#
#         MainWindow.setCentralWidget(self.centralwidget)
#
#         # Menu
#         self.menubar = QtWidgets.QMenuBar(MainWindow)
#         self.Menue = QtWidgets.QMenu("Menu", self.menubar)
#         MainWindow.setMenuBar(self.menubar)
#
#         self.actionLoad = QtWidgets.QAction("Load Base Folder", MainWindow)
#         self.Menue.addAction(self.actionLoad)
#         self.actionLoad.triggered.connect(self.load_base_folder)
#
#         self.actionShowExample = QtWidgets.QAction("Show Examples", MainWindow)
#         self.Menue.addAction(self.actionShowExample)
#         self.actionShowExample.triggered.connect(self.load_example_images)
#
#         self.menubar.addAction(self.Menue.menuAction())
#         self.statusbar = QtWidgets.QStatusBar(MainWindow)
#         MainWindow.setStatusBar(self.statusbar)
#
#         # Connect
#         self.pushButton_6.clicked.connect(self.show_previous)
#         self.pushButton_7.clicked.connect(self.show_next)
#         self.pushButton_2.clicked.connect(lambda: self.rate_image(1))
#         self.pushButton_4.clicked.connect(lambda: self.rate_image(0))
#         self.pushButton_5.clicked.connect(self.save_ratings)
#         self.pushButton_fullFrame.clicked.connect(self.show_full_frame)
#
#         # Shortcuts
#         QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), MainWindow,
#                             activated=lambda: self.rate_image(1))
#         QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), MainWindow,
#                             activated=lambda: self.rate_image(0))
#         QtWidgets.QShortcut(QtGui.QKeySequence("5"), MainWindow,
#                             activated=lambda: self.rate_image(2))
#
#         # State
#         self.base_folder = None
#         self.test_folders = []
#         self.current_test_index = -1
#         self.image_files = []
#         self.current_index = -1
#
#     # === Folder navigation ===
#     def load_base_folder(self):
#         folder = QtWidgets.QFileDialog.getExistingDirectory(
#             None, "Select Base Folder", "", QtWidgets.QFileDialog.ShowDirsOnly
#         )
#         if folder:
#             self.base_folder = folder
#             self.test_folders = [os.path.join(folder, f) for f in os.listdir(folder)
#                                  if f.startswith("test_") and os.path.isdir(os.path.join(folder, f))]
#             self.test_folders.sort()
#             if self.test_folders:
#                 self.current_test_index = 0
#                 self.load_test_folder()
#
#     def load_test_folder(self):
#         if 0 <= self.current_test_index < len(self.test_folders):
#             test_folder = self.test_folders[self.current_test_index]
#             extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#             self.image_files = sorted(
#                 [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(extensions)]
#             )
#             if self.image_files:
#                 self.current_index = 0
#                 self.show_image()
#
#     # === Image display and navigation ===
#     def show_previous(self):
#         if self.image_files and self.current_index > 0:
#             # normal case: go back one image
#             self.current_index -= 1
#             self.show_image()
#         elif self.current_test_index > 0:
#             # move to previous test folder
#             self.current_test_index -= 1
#             self.load_test_folder()
#             if self.image_files:
#                 self.current_index = len(self.image_files) - 1  # jump to last image
#                 self.show_image()
#
#     def show_next(self):
#         if self.image_files and self.current_index < len(self.image_files) - 1:
#             # normal case: go forward one image
#             self.current_index += 1
#             self.show_image()
#         elif self.current_test_index < len(self.test_folders) - 1:
#             # move to next test folder
#             self.current_test_index += 1
#             self.load_test_folder()
#             if self.image_files:
#                 self.current_index = 0  # start from first image
#                 self.show_image()
#
#     def show_image(self):
#         if 0 <= self.current_index < len(self.image_files):
#             img_path = self.image_files[self.current_index]
#             pixmap = QtGui.QPixmap(img_path)
#             scene = QtWidgets.QGraphicsScene()
#             scene.addPixmap(pixmap)
#             self.graphicsView.setScene(scene)
#             self.lineEdit_filename.setText(os.path.basename(img_path))
#
#     # === Ratings ===
#     def rate_image(self, score):
#         if 0 <= self.current_index < len(self.image_files):
#             img_path = self.image_files[self.current_index]
#             filename = os.path.basename(img_path)
#
#             # Save rating with folder context (better traceability)
#             test_folder_name = os.path.basename(self.test_folders[self.current_test_index])
#             self.ratings[f"{test_folder_name}/{filename}"] = score
#             print(f"Rated {test_folder_name}/{filename} as {score}")
#
#             # Move to next image (or next folder if at end)
#             self.show_next()
#
#     def load_example_images(self):
#         self.example_window = ExampleWindow(parent=None)
#         self.example_window.exec_()
#
#     def show_full_frame(self):
#         if not self.test_folders or self.current_index < 0:
#             QtWidgets.QMessageBox.warning(None, "Error", "No image loaded!")
#             return
#         current_filename = os.path.basename(self.image_files[self.current_index])
#         test_folder = self.test_folders[self.current_test_index]
#         full_frame_folder = os.path.join(test_folder, "full_frame")
#         full_frame_path = os.path.join(full_frame_folder, current_filename)
#         if os.path.exists(full_frame_path):
#             self.full_frame_window = FullFrameWindow(full_frame_path, parent=None)
#             self.full_frame_window.exec_()
#         else:
#             QtWidgets.QMessageBox.warning(None, "Error",
#                                           f"Full frame does not exist for:\n{current_filename}")
#
#     # === Save results ===
#     def save_ratings(self):
#         if not self.ratings:
#             QtWidgets.QMessageBox.warning(None, "Warning", "No ratings to save!")
#             return
#         first_name = self.lineEdit_FirstName.text().strip()
#         last_name = self.lineEdit_LastName.text().strip()
#         if not first_name or not last_name:
#             QtWidgets.QMessageBox.warning(None, "Warning", "Please enter both first and last name!")
#             return
#         wb = Workbook()
#         ws = wb.active
#         ws.title = "Image Ratings"
#         ws.append(["Image Name", "Score"])
#         for filename, score in self.ratings.items():
#             ws.append([filename, score])
#         save_path = os.path.join(self.base_folder, f"{first_name}_{last_name}.xlsx")
#         wb.save(save_path)
#         QtWidgets.QMessageBox.information(None, "Saved", f"Ratings saved to:\n{save_path}")
#
#
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
from PyQt5 import QtCore, QtWidgets, QtGui
from openpyxl import Workbook
import os
import random


class ExampleWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Example Images")
        self.resize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)

        label_examples = QtWidgets.QLabel("<b>Good Detection</b>")
        layout.addWidget(label_examples, alignment=QtCore.Qt.AlignCenter)

        mixed_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(mixed_layout)

        mixed_images = [
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good1.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good2.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good3.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good4.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\good6.png",
        ]

        for img_path in mixed_images:
            lbl = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(img_path).scaled(200, 200, QtCore.Qt.KeepAspectRatio)
            lbl.setPixmap(pixmap)
            mixed_layout.addWidget(lbl)

        label_bad = QtWidgets.QLabel("<b>Bad Detection</b>")
        layout.addWidget(label_bad, alignment=QtCore.Qt.AlignCenter)
        bad_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(bad_layout)
        bad_images = [
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad1.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad2.png",
            r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\Examples\Example2\bad3.png",
        ]
        for img_path in bad_images:
            lbl = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(img_path).scaled(200, 200, QtCore.Qt.KeepAspectRatio)
            lbl.setPixmap(pixmap)
            bad_layout.addWidget(lbl)


class FullFrameWindow(QtWidgets.QDialog):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Frame")
        self.resize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(img_path).scaled(750, 550, QtCore.Qt.KeepAspectRatio)
        lbl.setPixmap(pixmap)
        layout.addWidget(lbl, alignment=QtCore.Qt.AlignCenter)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(650, 480)
        self.ratings = {}
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # GraphicsView
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(220, 0, 400, 370))

        # Filename display
        self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filename.setGeometry(QtCore.QRect(220, 380, 400, 20))
        self.lineEdit_filename.setReadOnly(True)

        # Name fields
        self.lineEdit_FirstName = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_FirstName.setGeometry(QtCore.QRect(80, 10, 91, 20))
        self.lineEdit_LastName = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_LastName.setGeometry(QtCore.QRect(80, 40, 91, 20))

        # Labels
        self.label = QtWidgets.QLabel("Name", self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 60, 20))
        self.label_2 = QtWidgets.QLabel("Last Name", self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 60, 20))

        # Buttons
        self.pushButton_2 = QtWidgets.QPushButton("Good", self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(150, 140, 61, 23))
        self.pushButton_4 = QtWidgets.QPushButton("Bad", self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 140, 61, 23))
        self.pushButton_5 = QtWidgets.QPushButton("Save", self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 340, 161, 31))
        self.pushButton_6 = QtWidgets.QPushButton("Previous", self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 220, 75, 23))
        self.pushButton_7 = QtWidgets.QPushButton("Next", self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(120, 220, 75, 23))
        self.pushButton_fullFrame = QtWidgets.QPushButton("Show Full Frame", self.centralwidget)
        self.pushButton_fullFrame.setGeometry(QtCore.QRect(65, 275, 120, 30))

        # Horizontal lines
        for geom in [(10, 120, 200, 20), (10, 170, 200, 20), (10, 240, 200, 20), (10, 200, 200, 20), (10, 320, 200, 20)]:
            line = QtWidgets.QFrame(self.centralwidget)
            line.setGeometry(QtCore.QRect(*geom))
            line.setFrameShape(QtWidgets.QFrame.HLine)

        MainWindow.setCentralWidget(self.centralwidget)

        # Menu
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.Menue = QtWidgets.QMenu("Menu", self.menubar)
        MainWindow.setMenuBar(self.menubar)

        self.actionLoad = QtWidgets.QAction("Load Base Folder", MainWindow)
        self.Menue.addAction(self.actionLoad)
        self.actionLoad.triggered.connect(self.load_base_folder)

        self.actionShowExample = QtWidgets.QAction("Show Examples", MainWindow)
        self.Menue.addAction(self.actionShowExample)
        self.actionShowExample.triggered.connect(self.load_example_images)

        self.actionShuffle = QtWidgets.QAction("Reshuffle Order", MainWindow)
        self.Menue.addAction(self.actionShuffle)
        self.actionShuffle.triggered.connect(self.shuffle_playlist)

        self.menubar.addAction(self.Menue.menuAction())
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # Connect
        self.pushButton_6.clicked.connect(self.show_previous)
        self.pushButton_7.clicked.connect(self.show_next)
        self.pushButton_2.clicked.connect(lambda: self.rate_image(1))
        self.pushButton_4.clicked.connect(lambda: self.rate_image(0))
        self.pushButton_5.clicked.connect(self.save_ratings)
        self.pushButton_fullFrame.clicked.connect(self.show_full_frame)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), MainWindow,
                            activated=lambda: self.rate_image(1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), MainWindow,
                            activated=lambda: self.rate_image(0))
        QtWidgets.QShortcut(QtGui.QKeySequence("5"), MainWindow,
                            activated=lambda: self.rate_image(2))
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), MainWindow,
                            activated=self.shuffle_playlist)

        # State
        self.base_folder = None
        self.test_folders = []
        self.items = []          # shuffled playlist: list of dicts {path, filename, test_folder}
        self.current_index = -1

    # === Folder load & playlist build ===
    def load_base_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Base Folder", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        self.base_folder = folder
        self.test_folders = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.startswith("test_") and os.path.isdir(os.path.join(folder, f))
        ]
        self.test_folders.sort()  # only for stable references; display order will be random

        # Build playlist (all images across all test_* folders)
        self.build_playlist()
        if self.items:
            self.current_index = 0
            self.show_image()

    def build_playlist(self, seed=None):
        """Collect every image in every test_* folder and shuffle into a single list."""
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        items = []
        for test_folder in self.test_folders:
            for f in os.listdir(test_folder):
                if f.lower().endswith(exts):
                    items.append({
                        "path": os.path.join(test_folder, f),
                        "filename": f,
                        "test_folder": test_folder
                    })
        if not items:
            QtWidgets.QMessageBox.warning(None, "No Images", "No images were found in test_* folders.")
            self.items = []
            return

        rng = random.Random(seed)  # allow deterministic shuffle if you pass a seed
        rng.shuffle(items)
        self.items = items
        self.current_index = 0

    def shuffle_playlist(self):
        """Reshuffle current playlist (keeps ratings dict intact)."""
        if not self.items:
            return
        # Optional: keep current item at front when reshuffling to avoid jump
        current = self.items[self.current_index] if 0 <= self.current_index < len(self.items) else None
        rng = random.Random()
        rng.shuffle(self.items)
        if current:
            # Move current back to index 0 to keep continuity
            i = self.items.index(current)
            self.items[0], self.items[i] = self.items[i], self.items[0]
            self.current_index = 0
        self.show_image()

    # === Image display and navigation over shuffled list ===
    def show_previous(self):
        if self.items and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next(self):
        if self.items and self.current_index < len(self.items) - 1:
            self.current_index += 1
            self.show_image()

    def show_image(self):
        if 0 <= self.current_index < len(self.items):
            item = self.items[self.current_index]
            pixmap = QtGui.QPixmap(item["path"])
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            # Show both folder + file to keep traceability visible
            rel = f'{os.path.basename(item["test_folder"])}/{item["filename"]}'
            # self.lineEdit_filename.setText(rel)
            self.statusbar.showMessage(f'{self.current_index + 1} / {len(self.items)}')

    # === Ratings (keeps folder context) ===
    def rate_image(self, score):
        if not (0 <= self.current_index < len(self.items)):
            return
        item = self.items[self.current_index]
        key = f'{os.path.basename(item["test_folder"])}/{item["filename"]}'
        self.ratings[key] = score
        print(f"Rated {key} as {score}")
        self.show_next()

    def load_example_images(self):
        self.example_window = ExampleWindow(parent=None)
        self.example_window.exec_()

    def show_full_frame(self):
        if not (0 <= self.current_index < len(self.items)):
            QtWidgets.QMessageBox.warning(None, "Error", "No image loaded!")
            return
        item = self.items[self.current_index]
        full_frame_folder = os.path.join(item["test_folder"], "full_frame")
        full_frame_path = os.path.join(full_frame_folder, item["filename"])
        if os.path.exists(full_frame_path):
            self.full_frame_window = FullFrameWindow(full_frame_path, parent=None)
            self.full_frame_window.exec_()
        else:
            QtWidgets.QMessageBox.warning(None, "Error",
                                          f"Full frame does not exist for:\n{item['filename']}")

    # === Save results ===
    def save_ratings(self):
        if not self.ratings:
            QtWidgets.QMessageBox.warning(None, "Warning", "No ratings to save!")
            return
        first_name = self.lineEdit_FirstName.text().strip()
        last_name = self.lineEdit_LastName.text().strip()
        if not first_name or not last_name:
            QtWidgets.QMessageBox.warning(None, "Warning", "Please enter both first and last name!")
            return
        wb = Workbook()
        ws = wb.active
        ws.title = "Image Ratings"
        ws.append(["Image Name", "Score"])
        for filename, score in self.ratings.items():
            ws.append([filename, score])
        save_path = os.path.join(self.base_folder, f"{first_name}_{last_name}.xlsx")
        wb.save(save_path)
        QtWidgets.QMessageBox.information(None, "Saved", f"Ratings saved to:\n{save_path}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
