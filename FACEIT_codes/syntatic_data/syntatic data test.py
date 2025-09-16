# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error
# import os
# import glob
# """ --------------------------------------------------- Functions ---------------------------------- """
# def compute_center_errors(gt, pred):
#     return np.linalg.norm(gt - pred, axis=1)
#
# """++++++++++++++++++++++++++++++++++++++             Load Data                  +++++++++++++++++++++++++"""
# """ --------------------------------------------------- Base path ---------------------------------- """
# base_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\8.contrast_change_shadow\5"
#
# # 1. Find .csv file in base_path
# csv_files = glob.glob(os.path.join(base_path, "*.csv"))
# assert len(csv_files) > 0, "No CSV file found in base_path"
# csv_path = csv_files[0]
#
# # 2. Find FaceIt npz file
# faceit_folder = os.path.join(base_path, "FaceIt")
# faceit_files = glob.glob(os.path.join(faceit_folder, "faceit.npz"))
# assert len(faceit_files) > 0, "No faceit.npz found in FaceIt folder"
# faceIt_path = faceit_files[0]
#
# # 3. Find *_proc.npy file in base_path
# proc_files = glob.glob(os.path.join(base_path, "*proc.npy"))
# assert len(proc_files) > 0, "No *_proc.npy file found in base_path"
# facemap_path = proc_files[0]
#
# # 4. Define output directories
# result_dir = os.path.join(base_path, "result2")
# os.makedirs(result_dir, exist_ok=True)
#
# figure_dir = os.path.join(result_dir, "figures")
# os.makedirs(figure_dir, exist_ok=True)
#
# excel_path = os.path.join(result_dir, "pupil_detection_results.xlsx")
# """ ---------------------------------------------------Ground_trouth ---------------------------------- """
#
# df = pd.read_csv(csv_path)
# frame = df['frame_index']
# GT_area = df['pupil_area']
# GT_X = df['center_x']
# GT_Y = df['center_y']
# GT_Y_norm = GT_Y - GT_Y[0]
# GT_X_norm = GT_X - GT_X[0]
# """ ---------------------------------------------------FaceIt-----------------------------------------------------"""
# faceit = np.load(faceIt_path, allow_pickle=True)
# print("Available keys:", list(faceit.keys()))
# faceIt_pupil_dialtion = faceit['pupil_dilation']
# width_faceit = faceit['width']
# height_faceit = faceit['height']
# faceit_X = faceit['pupil_center_X']
# faceit_Y = faceit['pupil_center_y']
# faceit_Y_norm = faceit_Y - faceit_Y[0]
# faceit_X_norm = faceit_X - faceit_X[0]
# """ ---------------------------------------------------FaceMap-----------------------------------------------------"""
# facemap_data = np.load(facemap_path, allow_pickle=True)
# print("facemap key", facemap_data.item().get('pupil', [{}])[0].keys())
# facemap_pupil = facemap_data.item().get('pupil', [{}])[0].get('area', np.array([]))
# facemap_center = facemap_data.item().get('pupil', [{}])[0].get('com', np.array([]))
# facemap_X = facemap_center[:, 0]
# facemap_Y = facemap_center[:, 1]
# facemap_Y_norm = facemap_Y - facemap_Y[0]
# facemap_X_norm = facemap_X - facemap_X[0]
#
#
# """ ---------------------------------------------------Data organization-----------------------------------------------------"""
# # --- Stack GT and prediction coordinates into (N, 2) arrays ---
# gt_coords = np.stack((GT_X_norm, GT_Y_norm), axis=1)
# faceit_coords = np.stack((faceit_X_norm, faceit_Y_norm), axis=1)
# facemap_coords = np.stack((facemap_X_norm, facemap_Y_norm), axis=1)
#
#
#
# """ --------------------------------- FaceMap VS GT Error ------------------------------"""
# r2_facemap = r2_score(GT_area, facemap_pupil)
# rmse_facemap = mean_squared_error(GT_area, facemap_pupil, squared=False)
#
#
# print(f"facemap VS GT R² Score: {r2_facemap:.3f}")
# print(f"facemap VS GT RMSE: {rmse_facemap:.2f} pixels²")
#
#
# # Frame-by-frame absolute error
# area_errors_facemap_Gt = np.abs(GT_area - facemap_pupil)
#
#
# """ --------------------------------- FaceIt VS GT Error ------------------------------"""
# # absolut error for radius
# r2_faceit= r2_score(GT_area, faceIt_pupil_dialtion)
# rmse_faceit = mean_squared_error(GT_area, faceIt_pupil_dialtion, squared=False)
# print(f"FaceIT VS GT R² Score: {rmse_faceit:.3f}")
# print(f"FaceIT VS GT RMSE: {rmse_faceit:.2f} pixels²")
# # Frame-by-frame absolute error
# area_errors_faceit_Gt = np.abs(GT_area - faceIt_pupil_dialtion)
#
# # --- Compute Euclidean distance frame by frame ---
#
#
# faceit_center_error = compute_center_errors(gt_coords, faceit_coords)
# facemap_center_error = compute_center_errors(gt_coords, facemap_coords)
# # -------------- Calculating std ----------------------
# std_center_error_faceIt = np.std(faceit_center_error)
# std_center_error_facemap = np.std(facemap_center_error)
#
# std_area_error_faceit = np.std(area_errors_faceit_Gt)
# std_area_error_facemap = np.std(area_errors_facemap_Gt)
#
# # Compute percent error relative to GT area (Frame-by-frame)
# percent_error_facemap = (area_errors_facemap_Gt / GT_area) * 100
# percent_error_faceit = (area_errors_faceit_Gt / GT_area) * 100
#
# # Mean percent error
# mean_percent_error_facemap = np.mean(percent_error_facemap)
# mean_percent_error_faceit = np.mean(percent_error_faceit)
#
# print(f"FaceMap - Mean Percent Area Error: {mean_percent_error_facemap:.2f}%")
# print(f"FaceIt  - Mean Percent Area Error: {mean_percent_error_faceit:.2f}%")
#
#
# # --- Print mean error or plot ---
# print(f"FaceIt - Mean Center Error: {faceit_center_error.mean():.2f} pixels")
# print(f"FaceIt - Std Center Error: {std_center_error_faceIt.mean():.2f} pixel")
# print(f"FaceMap - Mean Center Error: {facemap_center_error.mean():.2f} pixels")
# print(f"FaceMap - Std Center Error: {std_center_error_facemap.mean():.2f} pixel")
# """------------------------------------------- Plot -----------------------------------------------------"""
#
# # ---------- Plot 4: Percent error (FaceMap) ----------
# plt.figure(figsize=(10, 4))
# plt.plot(frame, percent_error_facemap, color='teal', linewidth=1.5, label='FaceMap % Error')
# plt.plot(frame, percent_error_faceit, color='purple', linewidth=1.5, label='FaceIt % Error')
# plt.xlabel("Frame Number", fontsize=12)
# plt.ylabel("Percent Error (%)", fontsize=12)
# plt.title("Percent Area Error Over Time", fontsize=14)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.png'), dpi=300)
#
#
# # ---------- Plot 1: Center error ----------
# plt.figure(figsize=(10, 4))
# plt.plot(faceit_center_error, label='FaceIt', linewidth=2, color='firebrick')
# plt.plot(facemap_center_error, label='FaceMap', linewidth=2, color='gold')
# plt.xlabel('Frame')
# plt.ylabel('Euclidean center error (pixels)')
# plt.title('Euclidean Distance to Ground Truth Center')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'center_error_comparison.png'), dpi=300)
#
#
# # ---------- Plot 2: Area error (FaceIt) ----------
# plt.figure(figsize=(10, 4))
# plt.plot(frame, area_errors_faceit_Gt, color='yellow', linewidth=1.5, label='faceit')
# plt.plot(frame, area_errors_facemap_Gt, color='crimson', linewidth=1.5, label='facemap')
# plt.xlabel("Frame Number", fontsize=12)
# plt.ylabel("Absolute Error (pixels²)", fontsize=12)
# plt.title("Frame-by-Frame Absolute Area Error: FaceIt vs GT", fontsize=14)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'area_error.png'), dpi=300)
# plt.savefig(os.path.join(figure_dir, 'area_error.svg'), bbox_inches='tight')
#
#
# # ---------- Plot 5: pupil area trace  ----------
# plt.figure(figsize=(10, 4))
# plt.plot(frame, facemap_pupil, color='teal', linewidth=1.5, label='FaceMap')
# plt.plot(frame, faceIt_pupil_dialtion, color='purple', linewidth=1.5, label='FaceIt')
# plt.plot(frame, GT_area, color='gold', linewidth=1.5, label='Ground Truth', linestyle=':')
# plt.xlabel("Frame Number", fontsize=12)
# plt.ylabel("pupil area (pixel)", fontsize=12)
# plt.title("Comparing pupil area among pipelines", fontsize=14)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.png'), dpi=300)
# plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.svg'), bbox_inches='tight')
#
# # ---------- Plot 6: Trajectory with time as color ----------
# plt.figure(figsize=(10, 4))
# plt.plot(frame, facemap_center_error, color='teal', linewidth=1.5, label='FaceMap')
# plt.plot(frame, faceit_center_error, color='purple', linewidth=1.5, label='FaceIt')
# plt.xlabel("Frame Number", fontsize=12)
# plt.ylabel("center Euclidean error (pixel)", fontsize=12)
# plt.title("Comparing center Euclidean error among pipelines", fontsize=14)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figure_dir, 'Center_Euclidean_error_comparison.svg'), bbox_inches='tight')
# plt.savefig(os.path.join(figure_dir, 'Center_Euclidean_error_comparison.png'), dpi=300)
#
#
#
# print(f"✅ All figures saved to: {figure_dir}")
#
# # Save all summary metrics
# results = {
#     'Metric': [
#         'Mean Center Error',
#         'Std Center Error',
#         'R2 (Area)',
#         'RMSE (Area)',
#         'Std Area Error',
#         'Mean % Area Error'  # ✅ New metric
#     ],
#     'FaceIt': [
#         faceit_center_error.mean(),
#         std_center_error_faceIt,
#         r2_faceit,
#         rmse_faceit,
#         std_area_error_faceit,
#         mean_percent_error_faceit  # ✅ New value
#     ],
#     'FaceMap': [
#         facemap_center_error.mean(),
#         std_center_error_facemap,
#         r2_facemap,
#         rmse_facemap,
#         std_area_error_facemap,
#         mean_percent_error_facemap  # ✅ New value
#     ]
# }
#
# df_results = pd.DataFrame(results)
#
# # --- Per-frame error data ---
# df_errors = pd.DataFrame({
#     'Frame': frame,
#     'GT Area': GT_area,
#     'FaceIt Area': faceIt_pupil_dialtion,
#     'FaceMap Area': facemap_pupil,
#     'FaceIt Area Error': area_errors_faceit_Gt,
#     'FaceMap Area Error': area_errors_facemap_Gt,
#     'FaceIt % Area Error': percent_error_faceit,
#     'FaceMap % Area Error': percent_error_facemap,
#     'FaceIt Center Error': faceit_center_error,
#     'FaceMap Center Error': facemap_center_error
# })
#
#
# # --- Save everything to Excel ---
# with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
#     df_results.to_excel(writer, index=False, sheet_name="Summary")
#     df_errors.to_excel(writer, index=False, sheet_name="Per Frame Errors")
#
# print(f"✅ Results and per-frame data saved to:\n{excel_path}")
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os, glob, re

plt.rcParams["figure.dpi"] = 120

# --------------------------- Helpers ---------------------------

def compute_center_errors(gt, pred):
    return np.linalg.norm(gt - pred, axis=1)

def normalize_columns(df):
    """lowercase + remove units in () + unify separators"""
    def norm(s):
        s = str(s).lower()
        s = re.sub(r"\(.*?\)", "", s)        # remove units
        s = re.sub(r"[^a-z0-9]+", "_", s)    # non-alnum -> _
        return s.strip("_")
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

def load_dlc_excel(base_path):
    """
    Look for DLC/DLS Excel at:
      base_path/DLS/EyeTracking_Data_pupil_data.xlsx
      base_path/DLC/EyeTracking_Data_pupil_data.xlsx
      or any *pupil*data*.xlsx under DLS/DLC/base_path.
    Returns: DataFrame with columns: frame, pupil_area, x_center, y_center
    or None if not found.
    """
    candidates = [
        os.path.join(base_path, "DLS", "EyeTracking_Data_pupil_data.xlsx"),
        os.path.join(base_path, "DLC", "EyeTracking_Data_pupil_data.xlsx"),
    ]
    # fallback search
    for sub in ["DLS", "DLC", ""]:
        candidates += glob.glob(os.path.join(base_path, sub, "*pupil*data*.xlsx"))

    xl_path = next((p for p in candidates if os.path.exists(p)), None)
    if xl_path is None:
        print("⚠️ DLC Excel not found; skipping DLC.")
        return None

    print(f"🔎 DLC Excel: {xl_path}")
    df = pd.read_excel(xl_path)
    df = normalize_columns(df)

    # map likely column names
    col_map = {
        "frame": ["frame", "frame_index", "frames"],
        "pupil_area": ["pupil_area", "area", "pupilarea"],
        "x_center": ["x_center", "center_x", "x"],
        "y_center": ["y_center", "center_y", "y"],
    }
    out = {}
    for key, options in col_map.items():
        found = None
        for c in options:
            if c in df.columns:
                found = c
                break
        if found is None:
            print(f"⚠️ DLC column for '{key}' not found; available: {list(df.columns)}")
            out[key] = None
        else:
            out[key] = df[found].astype(float if key != "frame" else int).values

    # Build clean DataFrame if we at least have frame + centers or area
    if out["frame"] is None:
        print("⚠️ DLC has no 'frame' column; cannot align — skipping DLC.")
        return None

    return pd.DataFrame({
        "frame": out["frame"],
        "pupil_area": out["pupil_area"],
        "x_center": out["x_center"],
        "y_center": out["y_center"],
    })

def align_on_frame(df_ref, df_other, cols):
    """
    Left-join df_other to df_ref on 'frame'.
    df_ref must have 'frame'. cols are columns to bring from df_other.
    Returns arrays aligned to df_ref['frame'].
    """
    if df_other is None:
        return [None for _ in cols]
    j = df_ref[["frame"]].merge(df_other[["frame"] + cols], on="frame", how="left")
    return [j[c].values if c in j.columns else None for c in cols]

# --------------------------- Base paths & outputs ---------------------------

base_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_8_contrast_change_shadow\series_5"

csv_files = glob.glob(os.path.join(base_path, "*.csv"))
assert csv_files, "No CSV file found in base_path"
csv_path = csv_files[0]

faceit_folder = os.path.join(base_path, "FaceIt")
faceit_files = glob.glob(os.path.join(faceit_folder, "faceit.npz"))
assert faceit_files, "No faceit.npz found in FaceIt folder"
faceIt_path = faceit_files[0]

proc_files = glob.glob(os.path.join(base_path, "*proc.npy"))
assert proc_files, "No *_proc.npy file found in base_path"
facemap_path = proc_files[0]

result_dir = os.path.join(base_path, "result2")
figure_dir = os.path.join(result_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
excel_path = os.path.join(result_dir, "pupil_detection_results.xlsx")

# --------------------------- Ground Truth ---------------------------

gt_df = pd.read_csv(csv_path)
gt_df = gt_df.rename(columns={
    'frame_index': 'frame', 'pupil_area': 'pupil_area',
    'center_x': 'x_center', 'center_y': 'y_center'
})
frame = gt_df['frame'].values.astype(int)
GT_area = gt_df['pupil_area'].values.astype(float)
GT_X = gt_df['x_center'].values.astype(float)
GT_Y = gt_df['y_center'].values.astype(float)

GT_X_norm = GT_X - GT_X[0]
GT_Y_norm = GT_Y - GT_Y[0]
gt_coords = np.stack((GT_X_norm, GT_Y_norm), axis=1)

# --------------------------- FaceIt ---------------------------

faceit = np.load(faceIt_path, allow_pickle=True)
faceIt_pupil_dilation = faceit['pupil_dilation'].astype(float)
faceit_X = faceit['pupil_center_X'].astype(float)
faceit_Y = faceit['pupil_center_y'].astype(float)
faceit_X_norm = faceit_X - faceit_X[0]
faceit_Y_norm = faceit_Y - faceit_Y[0]
faceit_coords = np.stack((faceit_X_norm, faceit_Y_norm), axis=1)

# --------------------------- FaceMap ---------------------------

facemap_data = np.load(facemap_path, allow_pickle=True)
facemap_pupil = facemap_data.item().get('pupil', [{}])[0].get('area', np.array([])).astype(float)
facemap_center = facemap_data.item().get('pupil', [{}])[0].get('com', np.array([])).astype(float)
facemap_X = facemap_center[:, 0]
facemap_Y = facemap_center[:, 1]
facemap_X_norm = facemap_X - facemap_X[0]
facemap_Y_norm = facemap_Y - facemap_Y[0]
facemap_coords = np.stack((facemap_X_norm, facemap_Y_norm), axis=1)

# --------------------------- DLC from Excel (DLS) ---------------------------

dlc_df = load_dlc_excel(base_path)
dlc_area, dlc_x, dlc_y = align_on_frame(
    gt_df[["frame"]].copy(), dlc_df, cols=["pupil_area", "x_center", "y_center"]
)
if dlc_x is not None and dlc_y is not None:
    dlc_X_norm = dlc_x - dlc_x[0]
    dlc_Y_norm = dlc_y - dlc_y[0]
    dlc_coords = np.stack((dlc_X_norm, dlc_Y_norm), axis=1)
else:
    dlc_coords = None

# --------------------------- Metrics ---------------------------

# Area R² / RMSE
r2_facemap = r2_score(GT_area, facemap_pupil[:len(GT_area)])
rmse_facemap = mean_squared_error(GT_area, facemap_pupil[:len(GT_area)], squared=False)

r2_faceit = r2_score(GT_area, faceIt_pupil_dilation[:len(GT_area)])
rmse_faceit = mean_squared_error(GT_area, faceIt_pupil_dilation[:len(GT_area)], squared=False)

if dlc_area is not None:
    r2_dlc = r2_score(GT_area, dlc_area[:len(GT_area)])
    rmse_dlc = mean_squared_error(GT_area, dlc_area[:len(GT_area)], squared=False)
else:
    r2_dlc = rmse_dlc = np.nan

print(f"FaceMap VS GT  R²: {r2_facemap:.3f} | RMSE: {rmse_facemap:.2f}")
print(f"FaceIt  VS GT  R²: {r2_faceit:.3f} | RMSE: {rmse_faceit:.2f}")
print(f"DLC     VS GT  R²: {r2_dlc:.3f} | RMSE: {rmse_dlc:.2f}")

# Frame-by-frame absolute area error
area_errors_facemap_Gt = np.abs(GT_area - facemap_pupil[:len(GT_area)])
area_errors_faceit_Gt  = np.abs(GT_area - faceIt_pupil_dilation[:len(GT_area)])
area_errors_dlc_Gt     = np.abs(GT_area - dlc_area[:len(GT_area)]) if dlc_area is not None else None

# Center errors
faceit_center_error  = compute_center_errors(gt_coords, faceit_coords)
facemap_center_error = compute_center_errors(gt_coords, facemap_coords)
dlc_center_error     = compute_center_errors(gt_coords, dlc_coords) if dlc_coords is not None else None

# STDs
std_center_error_faceIt  = float(np.std(faceit_center_error))
std_center_error_facemap = float(np.std(facemap_center_error))
std_center_error_dlc     = float(np.std(dlc_center_error)) if dlc_center_error is not None else np.nan

std_area_error_faceit  = float(np.std(area_errors_faceit_Gt))
std_area_error_facemap = float(np.std(area_errors_facemap_Gt))
std_area_error_dlc     = float(np.std(area_errors_dlc_Gt)) if area_errors_dlc_Gt is not None else np.nan

# Percent area error
# Avoid division by zero:
safe_GT = np.where(GT_area == 0, np.nan, GT_area)
percent_error_facemap = (area_errors_facemap_Gt / safe_GT) * 100
percent_error_faceit  = (area_errors_faceit_Gt / safe_GT) * 100
percent_error_dlc     = (area_errors_dlc_Gt / safe_GT) * 100 if area_errors_dlc_Gt is not None else None

mean_percent_error_facemap = np.nanmean(percent_error_facemap)
mean_percent_error_faceit  = np.nanmean(percent_error_faceit)
mean_percent_error_dlc     = np.nanmean(percent_error_dlc) if percent_error_dlc is not None else np.nan

print(f"FaceMap - Mean % Area Error: {mean_percent_error_facemap:.2f}%")
print(f"FaceIt  - Mean % Area Error: {mean_percent_error_faceit:.2f}%")
if percent_error_dlc is not None:
    print(f"DLC     - Mean % Area Error: {mean_percent_error_dlc:.2f}%")

print(f"FaceIt  - Mean Center Error: {faceit_center_error.mean():.2f} px | Std: {std_center_error_faceIt:.2f} px")
print(f"FaceMap - Mean Center Error: {facemap_center_error.mean():.2f} px | Std: {std_center_error_facemap:.2f} px")
if dlc_center_error is not None:
    print(f"DLC     - Mean Center Error: {dlc_center_error.mean():.2f} px | Std: {std_center_error_dlc:.2f} px")

# --------------------------- Plots ---------------------------

# 1) Percent area error
plt.figure(figsize=(10, 4))
plt.plot(frame, percent_error_facemap, linewidth=1.5, label='FaceMap % Error')
plt.plot(frame, percent_error_faceit,  linewidth=1.5, label='FaceIt % Error')
if percent_error_dlc is not None:
    plt.plot(frame, percent_error_dlc, linewidth=1.5, label='DLC % Error')
plt.xlabel("Frame"); plt.ylabel("Percent Error (%)"); plt.title("Percent Area Error Over Time")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'percent_area_error_comparison.png'), dpi=300)

# 2) Center error
plt.figure(figsize=(10, 4))
plt.plot(faceit_center_error,  label='FaceIt',  linewidth=2)
plt.plot(facemap_center_error, label='FaceMap', linewidth=2)
if dlc_center_error is not None:
    plt.plot(dlc_center_error, label='DLC', linewidth=2)
plt.xlabel('Frame'); plt.ylabel('Center Error (px)'); plt.title('Euclidean Distance to GT Center')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'center_error_comparison.png'), dpi=300)

# 3) Absolute area error
plt.figure(figsize=(10, 4))
plt.plot(frame, area_errors_faceit_Gt,  linewidth=1.5, label='FaceIt')
plt.plot(frame, area_errors_facemap_Gt, linewidth=1.5, label='FaceMap')
if area_errors_dlc_Gt is not None:
    plt.plot(frame, area_errors_dlc_Gt, linewidth=1.5, label='DLC')
plt.xlabel("Frame"); plt.ylabel("Absolute Error (px²)"); plt.title("Area Error per Frame")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'area_error_comparison.png'), dpi=300)

# 4) Pupil area traces
plt.figure(figsize=(10, 4))
plt.plot(frame, facemap_pupil[:len(frame)],        linewidth=1.2, label='FaceMap')
plt.plot(frame, faceIt_pupil_dilation[:len(frame)],linewidth=1.2, label='FaceIt')
if dlc_area is not None:
    plt.plot(frame, dlc_area[:len(frame)],         linewidth=1.2, label='DLC')
plt.plot(frame, GT_area[:len(frame)],              linewidth=1.2, label='Ground Truth', linestyle=':')
plt.xlabel("Frame"); plt.ylabel("Pupil Area (px²)"); plt.title("Pupil Area: Pipelines vs GT")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'area_traces_comparison.png'), dpi=300)

print(f"✅ All figures saved to: {figure_dir}")

# --------------------------- Excel Output ---------------------------

summary = {
    'Metric': [
        'Mean Center Error',
        'Std Center Error',
        'R2 (Area)',
        'RMSE (Area)',
        'Std Area Error',
        'Mean % Area Error'
    ],
    'FaceIt': [
        float(faceit_center_error.mean()),
        std_center_error_faceIt,
        float(r2_faceit),
        float(rmse_faceit),
        std_area_error_faceit,
        float(mean_percent_error_faceit)
    ],
    'FaceMap': [
        float(facemap_center_error.mean()),
        std_center_error_facemap,
        float(r2_facemap),
        float(rmse_facemap),
        std_area_error_facemap,
        float(mean_percent_error_facemap)
    ],
    'DLC': [
        float(dlc_center_error.mean()) if dlc_center_error is not None else np.nan,
        std_center_error_dlc,
        float(r2_dlc) if not np.isnan(r2_dlc) else np.nan,
        float(rmse_dlc) if not np.isnan(rmse_dlc) else np.nan,
        std_area_error_dlc,
        float(mean_percent_error_dlc) if not np.isnan(mean_percent_error_dlc) else np.nan
    ]
}
df_results = pd.DataFrame(summary)

per_frame = {
    'Frame': frame,
    'GT Area': GT_area,
    'FaceIt Area': faceIt_pupil_dilation[:len(frame)],
    'FaceMap Area': facemap_pupil[:len(frame)],
    'FaceIt Area Error': area_errors_faceit_Gt[:len(frame)],
    'FaceMap Area Error': area_errors_facemap_Gt[:len(frame)],
    'FaceIt % Area Error': percent_error_faceit[:len(frame)],
    'FaceMap % Area Error': percent_error_facemap[:len(frame)],
    'FaceIt Center Error': faceit_center_error[:len(frame)],
    'FaceMap Center Error': facemap_center_error[:len(frame)],
}
if dlc_area is not None:
    per_frame['DLC Area'] = dlc_area[:len(frame)]
    per_frame['DLC Area Error'] = area_errors_dlc_Gt[:len(frame)]
    per_frame['DLC % Area Error'] = percent_error_dlc[:len(frame)]
if dlc_center_error is not None:
    per_frame['DLC Center Error'] = dlc_center_error[:len(frame)]
df_errors = pd.DataFrame(per_frame)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_results.to_excel(writer, index=False, sheet_name="Summary")
    df_errors.to_excel(writer, index=False, sheet_name="Per Frame Errors")

print(f"✅ Results and per-frame data saved to:\n{excel_path}")
