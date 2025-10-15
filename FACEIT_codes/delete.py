import numpy as np
import matplotlib.pyplot as plt
data = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\faceit.npz", allow_pickle=True)
Face_frame = data['Face_frame']
print("Face_frame", Face_frame)
Pupil_frame = data['Pupil_frame']
print("Pupil_frame", Pupil_frame)

import matplotlib.patches as patches

import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Load the image ---
image = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\1710514480.552814.npy", allow_pickle=True)

# --- Face_frame coordinates ---
top, bottom, left, right = Pupil_frame

# --- Compute ellipse center and axes ---
center = (int((left + right) / 2), int((top + bottom) / 2))
axes = (int((right - left) / 2), int((bottom - top) / 2))  # (width/2, height/2)
angle = 0  # in degrees (no rotation)

# --- Copy image for drawing ---
image_with_ellipse = image.copy()

# --- Draw the ellipse outline directly on the image ---
# If the image is grayscale, convert to BGR so we can use color
if image_with_ellipse.ndim == 2:
    image_with_ellipse = cv2.cvtColor(image_with_ellipse, cv2.COLOR_GRAY2BGR)

cv2.ellipse(
    image_with_ellipse, center, axes, angle,
    0, 360, (0, 255, 0), 2  # green outline
)

# --- Show the result ---
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
plt.title("Elliptical ROI Overlay (No Mask)")
plt.axis('off')
plt.show()
