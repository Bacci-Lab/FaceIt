import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os
import glob
def compute_center_errors(gt, pred):
    return np.linalg.norm(gt - pred, axis=1)
""" --------------------------------------------------- Base path ---------------------------------- """
base_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_1"

# 1. Find .csv file in base_path
# Find the first CSV file
xlsx_files = glob.glob(os.path.join(base_path, "*.csv"))
assert len(xlsx_files) > 0, "No CSV file found in base_path"
xlsx_file = xlsx_files[0]

# 2. Find FaceIt npz file
faceit_folder = os.path.join(base_path, "FaceIt")
faceit_files = glob.glob(os.path.join(faceit_folder, "faceit.npz"))
assert len(faceit_files) > 0, "No faceit.npz found in FaceIt folder"
faceIt_path = faceit_files[0]

# 3. Find *_proc.npy file in base_path
proc_files = glob.glob(os.path.join(base_path, "*proc.npy"))
assert len(proc_files) > 0, "No *_proc.npy file found in base_path"
facemap_path = proc_files[0]

# 4. Define output directories
result_dir = os.path.join(base_path, "result3")
os.makedirs(result_dir, exist_ok=True)

figure_dir = os.path.join(result_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

excel_path = os.path.join(result_dir, "pupil_detection_results.xlsx")
""" ---------------------------------------------------Ground_trouth ---------------------------------- """

# Read CSV correctly (since it's not actually Excel)
df = pd.read_csv(xlsx_file)

# Print preview
print(df.head())
print("------------- Ground Truth ----------------")
frame = df['frame'].astype(int).values

GT_X = df['center_x']
GT_Y = df['center_y']
# Normalize center coordinates
GT_X_norm = GT_X - GT_X.iloc[0]
GT_Y_norm = GT_Y - GT_Y.iloc[0]

# Compute GT_area depending on available columns
if 'radius' in df.columns:
    print("Using 'radius' to compute GT_area...")
    GT_radius = df['radius']
    GT_width = GT_radius * 2
    GT_height = GT_radius * 2
    GT_area = (np.pi * GT_width * GT_height) / 4
elif 'pupil_area' in df.columns:
    print("Using 'pupil_area' from CSV directly...")
    GT_area = df['pupil_area']
else:
    raise ValueError("Neither 'radius' nor 'pupil_area' found in the CSV.")

""" ---------------------------------------------------FaceIt-----------------------------------------------------"""
faceit = np.load(faceIt_path, allow_pickle=True)
faceIt_pupil_dialtion = faceit['pupil_dilation']
width_faceit = faceit['width']
height_faceit = faceit['height']

faceit_X = faceit['pupil_center_X']
faceit_Y = faceit['pupil_center_y']
faceit_Y_norm = faceit_Y - faceit_Y[0]
faceit_X_norm = faceit_X - faceit_X[0]
""" ---------------------------------------------------FaceMap-----------------------------------------------------"""
facemap_data = np.load(facemap_path, allow_pickle=True)
facemap_pupil = facemap_data.item().get('pupil', [{}])[0].get('area', np.array([]))
facemap_center = facemap_data.item().get('pupil', [{}])[0].get('com', np.array([]))
facemap_X = facemap_center[:, 0]
facemap_Y = facemap_center[:, 1]

facemap_Y_norm = facemap_Y - facemap_Y[0]
facemap_X_norm = facemap_X - facemap_X[0]
############## select ###########
GT_area = GT_area
faceIt_pupil_dialtion = faceIt_pupil_dialtion
############################
plt.plot(frame, facemap_pupil, label= "facemap")
plt.plot(frame, GT_area, label = "manual")
plt.plot(frame, faceIt_pupil_dialtion, label = "faceIt")
plt.legend()
plt.show()


""" ---------------------------------------------------Data organization-----------------------------------------------------"""
# --- Stack GT and prediction coordinates into (N, 2) arrays ---
gt_coords = np.stack((GT_X_norm, GT_Y_norm), axis=1)
faceit_coords = np.stack((faceit_X_norm, faceit_Y_norm), axis=1)
facemap_coords = np.stack((facemap_X_norm, facemap_Y_norm), axis=1)



""" --------------------------------- FaceMap VS GT Error ------------------------------"""
r2_facemap = r2_score(GT_area, facemap_pupil)
rmse_facemap = mean_squared_error(GT_area, facemap_pupil, squared=False)


print(f"facemap VS GT R² Score: {r2_facemap:.3f}")
print(f"facemap VS GT RMSE: {rmse_facemap:.2f} pixels²")


# Frame-by-frame absolute error
area_errors_facemap_Gt = np.abs(GT_area - facemap_pupil)


""" --------------------------------- FaceIt VS GT Error ------------------------------"""
# absolut error for radius
r2_faceit= r2_score(GT_area, faceIt_pupil_dialtion)
rmse_faceit = mean_squared_error(GT_area, faceIt_pupil_dialtion, squared=False)
print(f"FaceIT VS GT R² Score: {rmse_faceit:.3f}")
print(f"FaceIT VS GT RMSE: {rmse_faceit:.2f} pixels²")
# Frame-by-frame absolute error
area_errors_faceit_Gt = np.abs(GT_area - faceIt_pupil_dialtion)

# --- Compute Euclidean distance frame by frame ---


faceit_center_error = compute_center_errors(gt_coords, faceit_coords)
facemap_center_error = compute_center_errors(gt_coords, facemap_coords)
# -------------- Calculating std ----------------------
std_center_error_faceIt = np.std(faceit_center_error)
std_center_error_facemap = np.std(facemap_center_error)

std_area_error_faceit = np.std(area_errors_faceit_Gt)
std_area_error_facemap = np.std(area_errors_facemap_Gt)

# Compute percent error relative to GT area (Frame-by-frame)
percent_error_facemap = (area_errors_facemap_Gt / GT_area) * 100
percent_error_faceit = (area_errors_faceit_Gt / GT_area) * 100

# Mean percent error
mean_percent_error_facemap = np.mean(percent_error_facemap)
mean_percent_error_faceit = np.mean(percent_error_faceit)

print(f"FaceMap - Mean Percent Area Error: {mean_percent_error_facemap:.2f}%")
print(f"FaceIt  - Mean Percent Area Error: {mean_percent_error_faceit:.2f}%")


# --- Print mean error or plot ---
print(f"FaceIt - Mean Center Error: {faceit_center_error.mean():.2f} pixels")
print(f"FaceIt - Std Center Error: {std_center_error_faceIt.mean():.2f} pixel")
print(f"FaceMap - Mean Center Error: {facemap_center_error.mean():.2f} pixels")
print(f"FaceMap - Std Center Error: {std_center_error_facemap.mean():.2f} pixel")
"""------------------------------------------- Plot -----------------------------------------------------"""

# ---------- Plot 4: Percent error (FaceMap) ----------
plt.figure(figsize=(10, 4))
plt.plot(frame, percent_error_facemap, color='teal', linewidth=1.5, label='FaceMap % Error')
plt.plot(frame, percent_error_faceit, color='purple', linewidth=1.5, label='FaceIt % Error')
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Percent Error (%)", fontsize=12)
plt.title("Percent Area Error Over Time", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.png'), dpi=300)


# ---------- Plot 1: Center error ----------
plt.figure(figsize=(10, 4))
plt.plot(faceit_center_error, label='FaceIt', linewidth=2, color='firebrick')
plt.plot(facemap_center_error, label='FaceMap', linewidth=2, color='gold')
plt.xlabel('Frame')
plt.ylabel('Euclidean center error (pixels)')
plt.title('Euclidean Distance to Ground Truth Center')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'center_error_comparison.png'), dpi=300)


# ---------- Plot 2: Area error (FaceIt) ----------

plt.figure(figsize=(10, 4))
plt.plot(frame, area_errors_faceit_Gt, color='yellow', linewidth=1.5, label='faceit')
plt.plot(frame, area_errors_facemap_Gt, color='crimson', linewidth=1.5, label='facemap')
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Absolute Error (pixels²)", fontsize=12)
plt.title("Frame-by-Frame Absolute Area Error: FaceIt vs GT", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'area_error.png'), dpi=300)
plt.savefig(os.path.join(figure_dir, 'area_error.svg'), bbox_inches='tight')

# -------------- check reverse ----------------
print("area_errors_faceit_Gt", area_errors_faceit_Gt)
accuracy_faceit = 1 - (area_errors_faceit_Gt /GT_area)
accuracy_facemap = 1 - (area_errors_facemap_Gt /GT_area)


plt.figure(figsize=(10, 4))
plt.plot(frame, accuracy_faceit, color='yellow', linewidth=1.5, label='FaceIt Accuracy')
plt.plot(frame, accuracy_facemap, color='crimson', linewidth=1.5, label='FaceMap Accuracy')
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Accuracy (1 = Perfect)", fontsize=12)
plt.title("Frame-by-Frame Accuracy of Pupil Area Detection", fontsize=14)
plt.legend()
# plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'area_accuracy.png'), dpi=300)


# ---------- Plot 5: pupil area trace  ----------
plt.figure(figsize=(10, 4))
plt.plot(frame, facemap_pupil, color='teal', linewidth=1.5, label='FaceMap')
plt.plot(frame, faceIt_pupil_dialtion, color='purple', linewidth=1.5, label='FaceIt')
plt.plot(frame, GT_area, color='gold', linewidth=1.5, label='Ground Truth', linestyle=':')
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("pupil area (pixel)", fontsize=12)
plt.title("Comparing pupil area among pipelines", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.png'), dpi=300)
plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.svg'), bbox_inches='tight')

# ---------- Plot 6: Trajectory with time as color ----------
plt.figure(figsize=(10, 4))
plt.plot(frame, facemap_center_error, color='teal', linewidth=1.5, label='FaceMap')
plt.plot(frame, faceit_center_error, color='purple', linewidth=1.5, label='FaceIt')
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("center Euclidean error (pixel)", fontsize=12)
plt.title("Comparing center Euclidean error among pipelines", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'Center_Euclidean_error_comparison.svg'), bbox_inches='tight')
plt.savefig(os.path.join(figure_dir, 'Center_Euclidean_error_comparison.png'), dpi=300)
plt.show()



print(f"✅ All figures saved to: {figure_dir}")

# Save all summary metrics
results = {
    'Metric': [
        'Mean Center Error',
        'Std Center Error',
        'R2 (Area)',
        'RMSE (Area)',
        'Std Area Error',
        'Mean % Area Error'  # ✅ New metric
    ],
    'FaceIt': [
        faceit_center_error.mean(),
        std_center_error_faceIt,
        r2_faceit,
        rmse_faceit,
        std_area_error_faceit,
        mean_percent_error_faceit  # ✅ New value
    ],
    'FaceMap': [
        facemap_center_error.mean(),
        std_center_error_facemap,
        r2_facemap,
        rmse_facemap,
        std_area_error_facemap,
        mean_percent_error_facemap  # ✅ New value
    ]
}

df_results = pd.DataFrame(results)

# --- Per-frame error data ---
df_errors = pd.DataFrame({
    'Frame': frame,
    'GT Area': GT_area,
    'FaceIt Area': faceIt_pupil_dialtion,
    'FaceMap Area': facemap_pupil,
    'FaceIt Area Error': area_errors_faceit_Gt,
    'FaceMap Area Error': area_errors_facemap_Gt,
    'FaceIt % Area Error': percent_error_faceit,
    'FaceMap % Area Error': percent_error_facemap,
    'FaceIt Center Error': faceit_center_error,
    'FaceMap Center Error': facemap_center_error
})


# --- Save everything to Excel ---
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_results.to_excel(writer, index=False, sheet_name="Summary")
    df_errors.to_excel(writer, index=False, sheet_name="Per Frame Errors")

print(f"✅ Results and per-frame data saved to:\n{excel_path}")
