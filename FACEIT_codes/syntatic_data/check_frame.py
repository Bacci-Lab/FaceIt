import cv2
import numpy as np
import matplotlib.pyplot as plt
faceIt_path = r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\faceit.npz"
faceit = np.load(faceIt_path, allow_pickle=True)
print("Available keys:", list(faceit.keys()))
frame_pos = faceit['frame_pos'][0]
frame_center = faceit['frame_center'][0]
frame_axes = faceit['frame_axes'][0]
print(frame_pos)
# ------------------------ Config ------------------------ #
video_path = r"C:\Users\faezeh.rabbani\Downloads\output3_web.mp4"
top, bottom, left, right = frame_pos              # Subregion bounds

pupil_center = (113, 65)                                     # Predicted pupil ellipse
pupil_axes = (28, 25)

gt_center = (120, 65)                                        # Ground truth pupil ellipse
gt_axes = (26, 27)

# ------------------------ Load & Crop Frame ------------------------ #
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read the video.")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
subregion_gray = gray[top:bottom, left:right]
height, width = subregion_gray.shape

# ------------------------ Create Final Image with White Outside ------------------------ #
# Step 1: Create filled outer ellipse mask
outer_mask = np.zeros_like(subregion_gray, dtype=np.uint8)
cv2.ellipse(outer_mask, center=frame_center, axes=frame_axes,
            angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

# Step 2: Invert to get outside
outer_mask_inv = cv2.bitwise_not(outer_mask)

# Step 3: Combine image with white background
inside = cv2.bitwise_and(subregion_gray, subregion_gray, mask=outer_mask)
outside = cv2.bitwise_and(np.full_like(subregion_gray, 255), 255, mask=outer_mask_inv)
final_image = cv2.add(inside, outside)

# ------------------------ Show Final Image ------------------------ #
plt.figure(figsize=(6, 4))
plt.imshow(subregion_gray, cmap='gray')
plt.axis('off')
plt.title("Outside White, Inside Preserved")
plt.show()

# ------------------------ Create Predicted Pupil Mask ------------------------ #
pupil_mask = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.ellipse(pupil_mask, center=pupil_center, axes=pupil_axes,
            angle=0, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=-1)

# ------------------------ Create Ground Truth Mask ------------------------ #
gt_mask = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.ellipse(gt_mask, center=gt_center, axes=gt_axes,
            angle=0, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=-1)

# ------------------------ Show Pupil & GT Masks ------------------------ #
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(cv2.cvtColor(pupil_mask, cv2.COLOR_BGR2RGB))
axs[0].set_title("Predicted Pupil Mask")
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB))
axs[1].set_title("Ground Truth Mask")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# ------------------------ Compute IoU ------------------------ #
# Convert both masks to binary
pupil_binary = (cv2.cvtColor(pupil_mask, cv2.COLOR_BGR2GRAY) == 0).astype(np.uint8)
gt_binary = (cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY) == 0).astype(np.uint8)

assert pupil_binary.shape == gt_binary.shape, "Mask size mismatch!"

intersection = np.logical_and(pupil_binary, gt_binary).sum()
union = np.logical_or(pupil_binary, gt_binary).sum()
iou_score = intersection / union if union > 0 else 0

print(f"IoU (Intersection over Union): {iou_score:.4f}")

# ------------------------ Visualize Overlap ------------------------ #
overlap_vis = np.zeros((height, width, 3), dtype=np.uint8)
overlap_vis[(pupil_binary == 1) & (gt_binary == 1)] = [0, 255, 0]     # TP: Green
overlap_vis[(pupil_binary == 1) & (gt_binary == 0)] = [255, 0, 0]     # FP: Red
overlap_vis[(pupil_binary == 0) & (gt_binary == 1)] = [0, 0, 255]     # FN: Blue

plt.figure(figsize=(6, 4))
plt.imshow(overlap_vis)
plt.axis('off')
plt.title(f"IoU = {iou_score:.3f} — Green: TP, Red: FP, Blue: FN")
plt.show()
