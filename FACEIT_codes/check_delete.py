# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.metrics import mean_squared_error, mean_absolute_error
# #
# # # --- 1) Load your data --------------------------------------
# # data = np.load(r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\FaceIt\faceit.npz")
# # pupil_dilation = data['pupil_dilation']           # detected area (likely raw)
# #
# # csv_path = r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\output_data.csv"
# # df = pd.read_csv(csv_path)                         # df has columns: frame_index, pupil_area (ground truth)
# #
# # # Make sure both are the same length:
# # N1 = len(df)
# # N2 = len(pupil_dilation)
# # assert N1 == N2, "Lengths differ: ground truth has {}, pipeline has {}".format(N1, N2)
# #
# # # --- 2) Zero‐center (or z-score) both traces ----------------
# # # If you’ve already zero‐centered ground truth:
# # df['gt_zero'] = df['pupil_area'] - df['pupil_area'].mean()
# # det_zero     = pupil_dilation - np.mean(pupil_dilation)
# #
# # # If you also want to match their amplitude (so the error metric is unitless),
# # # you can z‐score both. Uncomment these two lines to use z‐scores instead:
# # # df['gt_z']  = df['gt_zero'] / df['gt_zero'].std()
# # # det_z       = det_zero       / det_zero.std()
# # # Then for metrics use df['gt_z'] and det_z.
# #
# # # For now, let’s compute metrics on zero‐centered data (not z‐scored):
# # gt = df['gt_zero'].values
# # det = det_zero
# #
# # # --- 3) Compute Pearson correlation --------------------------
# # corrcoef = np.corrcoef(gt, det)[0,1]
# # print(f"Pearson r = {corrcoef:.4f}")
# #
# # # --- 4) Compute RMSE and MAE ----------------------------------
# # rmse = np.sqrt(mean_squared_error(gt, det))
# # mae  = mean_absolute_error(gt, det)
# # print(f"RMSE = {rmse:.2f}   (in same units as pupil_area)")
# # print(f"MAE  = {mae:.2f}")
# #
# # # If you used z‐scores instead, then RMSE is in “σ units”:
# # #    rmse_z = np.sqrt(mean_squared_error(df['gt_z'], det_z))
# # #    print("RMSE (z‐score) =", rmse_z)
# #
# # # 4) Plot the zero‐centered (or z‐scored) curves. Here I’ll plot z‐scores:
# # plt.figure(figsize=(10,5))
# # plt.plot(df["frame_index"], df['pupil_area'], label="Synthetic (z‐score)")
# # plt.plot(df["frame_index"], pupil_dilation,  label="FaceIt Detection  (z‐score)", alpha=0.7)
# # plt.xlabel("Frame")
# # plt.ylabel("Normalized area (σ units)")
# # plt.title("Pupil‐area traces (zero‐mean, unit‐variance)")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig(r"C:\Users\faezeh.rabbani\Desktop\Adriana\faezeh_test\area_zscore_comparison.png")
# # plt.show()
# #
# #
# # # --- 5) Scatter plot: GT vs Det -------------------------------
# # plt.figure(figsize=(5,5))
# # plt.scatter(gt, det, alpha=0.3, s=10)
# # plt.xlabel("Ground‐truth (zero‐mean) pupil area")
# # plt.ylabel("Detected (zero‐mean) pupil area")
# # plt.title("Scatter: GT vs Det (r = {:.3f})".format(corrcoef))
# # plt.axline((0,0), slope=1, color='k', linestyle='--', linewidth=0.7)  # unity line
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig(r"C:\Users\faezeh.rabbani\Desktop\Adriana\faezeh_test\scatter_gt_vs_det.png")
# # plt.show()
# #
# # # --- 6) Optional: Bland–Altman plot ---------------------------
# # # Bland–Altman shows bias (mean difference) and limits of agreement.
# # diff = det - gt
# # mean_of_two = (det + gt) / 2
# # bias = np.mean(diff)
# # loA = bias + 1.96 * np.std(diff)  # upper limit of agreement
# # loB = bias - 1.96 * np.std(diff)  # lower limit
# #
# # plt.figure(figsize=(6,4))
# # plt.scatter(mean_of_two, diff, alpha=0.3, s=10)
# # plt.axhline(bias,    color='red', linestyle='-')
# # plt.axhline(loA,     color='gray', linestyle='--', linewidth=0.8)
# # plt.axhline(loB,     color='gray', linestyle='--', linewidth=0.8)
# # plt.xlabel("Mean of GT and Det (zero‐mean)")
# # plt.ylabel("Det – GT")
# # plt.title("Bland–Altman (bias={:.2f})".format(bias))
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig(r"C:\Users\faezeh.rabbani\Desktop\Adriana\faezeh_test\bland_altman.png")
# # plt.show()
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# # === 1) FILE PATHS – EDIT THESE TO YOUR ACTUAL FILES ==============
# # 1. Ground truth:
# gt_csv       = r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\3\output_data.csv"
#
# # 2. FaceIt pipeline’s estimate (NPZ contains “pupil_dilation”):
# faceit_npz   = r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\3\FaceIt\faceit.npz"
#
# # 3. Facemap pipeline’s estimate (NPZ contains “pupil_dilation”):
# facemap_npz  = r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\3\7.2_proc.npy"
#
#
# # 4. “Other” pipeline’s estimate (CSV with columns: timecode (s), pupil-area):
# Meye_csv    = r"C:\Users\faezeh.rabbani\FACEIT_DATA\Synthetic_video_test\3\7.2_it.csv"
#
#
# # === 2) LOAD GROUND TRUTH FROM output_data.csv ===================
# df_gt = pd.read_csv(gt_csv)
# if "frame_index" not in df_gt.columns or "pupil_area" not in df_gt.columns:
#     raise KeyError(f"Expected 'frame_index' and 'pupil_area' in {gt_csv}, got: {df_gt.columns.tolist()}")
#
# gt_area = df_gt["pupil_area"].astype(float).values
# N_gt     = len(gt_area)
# duration_seconds = 15.0
# time_gt = np.linspace(0.0, duration_seconds, N_gt)
#
#
# # === 3) LOAD FaceIt PIPELINE FROM faceit_npz =====================
# data = np.load(faceit_npz, allow_pickle=True)
# if "pupil_dilation" not in data:
#     raise KeyError(f"Expected 'pupil_dilation' inside {faceit_npz}")
# det_faceit_raw = data["pupil_dilation"].astype(float)
# if len(det_faceit_raw) != N_gt:
#     raise ValueError(f"FaceIt length={len(det_faceit_raw)} but GT length={N_gt}")
#
#
# # === 4) LOAD Facemap PIPELINE FROM facemap_npz ===================
#
#
# data = np.load(facemap_npz, allow_pickle=True)
# det_facemap_raw = data.item().get('pupil', [{}])[0].get('area', np.array([]))
# if len(det_facemap_raw) != N_gt:
#     raise ValueError(f"Facemap length={len(det_facemap_raw)} but GT length={N_gt}")
#
#
# # === 5) LOAD “Meye” PIPELINE (timecode vs. pupil-area) ==========
# df_o = pd.read_csv(Meye_csv)
# if "timecode" not in df_o.columns or "pupil-area" not in df_o.columns:
#     raise KeyError(f"Expected 'timecode' & 'pupil-area' in {Meye_csv}, got: {df_o.columns.tolist()}")
#
# time_o = df_o["timecode"].astype(float).values
# area_o = df_o["pupil-area"].astype(float).values
#
# # Interpolate “Meye” onto uniform time_gt grid:
# det_Meye_interp = np.interp(time_gt, time_o, area_o)
#
#
# # === 6) LINEARLY‐SCALE “Meye” INTO GT UNITS (pixel²) ==============
# x       = det_Meye_interp
# y       = gt_area
# x_mean  = x.mean()
# y_mean  = y.mean()
# numer   = np.sum((x - x_mean) * (y - y_mean))
# denom   = np.sum((x - x_mean)**2)
# a_Meye = numer / denom
# b_Meye = y_mean - a_Meye * x_mean
#
# det_Meye_scaled = a_Meye * det_Meye_interp + b_Meye
#
#
# # === 7) ZERO‐CENTER ALL FOUR TRACES (GT, FaceIt, Facemap, Meye) ==
# gt_zero         = gt_area         - np.mean(gt_area)
# det_faceit_zero = det_faceit_raw  - np.mean(det_faceit_raw)
# det_facemap_zero= det_facemap_raw - np.mean(det_facemap_raw)
# det_Meye_zero  = det_Meye_scaled - np.mean(det_Meye_scaled)
#
#
# # === 8) COMPUTE METRICS FOR EACH PIPELINE =========================
# def compute_stats(gt, det):
#     r    = np.corrcoef(gt, det)[0, 1]
#     rmse = np.sqrt(mean_squared_error(gt, det))
#     mae  = mean_absolute_error(gt, det)
#     return r, rmse, mae
#
# r_fi,  rmse_fi,  mae_fi  = compute_stats(gt_zero, det_faceit_zero)
# r_fm,  rmse_fm,  mae_fm  = compute_stats(gt_zero, det_facemap_zero)
# r_oth, rmse_oth, mae_oth = compute_stats(gt_zero, det_Meye_zero)
#
# print("=== FaceIt pipeline vs GT ===")
# print(f"  Pearson r = {r_fi:.4f}")
# print(f"  RMSE       = {rmse_fi:.2f}")
# print(f"  MAE        = {mae_fi:.2f}\n")
#
# print("=== Facemap pipeline vs GT ===")
# print(f"  Pearson r = {r_fm:.4f}")
# print(f"  RMSE       = {rmse_fm:.2f}")
# print(f"  MAE        = {mae_fm:.2f}\n")
#
# print("=== Meye pipeline vs GT ===")
# print(f"  Pearson r = {r_oth:.4f}")
# print(f"  RMSE       = {rmse_oth:.2f}")
# print(f"  MAE        = {mae_oth:.2f}")
#
#
# # === 9) PLOT TIME‐SERIES (0→15 s) ==================================
# plt.figure(figsize=(10, 5))
# plt.plot(time_gt, gt_zero,           label="Ground Truth",              color="black", linewidth=1.5)
# plt.plot(time_gt, det_faceit_zero,   label=f"FaceIt (r={r_fi:.3f})",     color="red",  alpha=0.7, linewidth=1.5)
# plt.plot(time_gt, det_facemap_zero,  label=f"Facemap (r={r_fm:.3f})",    color="green", alpha=0.7, linewidth=1.5)
# plt.plot(time_gt, det_Meye_zero,    label=f"Meye (r={r_oth:.3f})",     color="orange",alpha=0.7, linewidth=1.5)
#
# plt.xlabel("Time (s)")
# plt.ylabel("Pupil Area (zero‐mean)")
# plt.title("Comparison Over 15 s")
# plt.legend(loc="upper right")
# plt.grid(True)
# plt.tight_layout()
#
# out_folder = os.path.dirname(gt_csv)
# plt.savefig(os.path.join(out_folder, "time_series_all_three.png"), dpi=150)
# plt.show()
#
#
# # === 10) SCATTER PLOTS =============================================
# # (a) GT vs FaceIt
# plt.figure(figsize=(5, 5))
# plt.scatter(gt_zero, det_faceit_zero, alpha=0.3, s=10, color="red",
#             label=f"FaceIt (r={r_fi:.3f})")
# plt.axline((0,0), slope=1, color="k", linestyle="--", linewidth=0.7)
# plt.xlabel("GT (zero‐mean)")
# plt.ylabel("FaceIt (zero‐mean)")
# plt.title("Scatter: GT vs FaceIt")
# plt.legend(loc="upper left")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "scatter_gt_vs_faceit.png"), dpi=150)
# plt.show()
#
# # (b) GT vs Facemap
# plt.figure(figsize=(5, 5))
# plt.scatter(gt_zero, det_facemap_zero, alpha=0.3, s=10, color="green",
#             label=f"Facemap (r={r_fm:.3f})")
# plt.axline((0,0), slope=1, color="k", linestyle="--", linewidth=0.7)
# plt.xlabel("GT (zero‐mean)")
# plt.ylabel("Facemap (zero‐mean)")
# plt.title("Scatter: GT vs Facemap")
# plt.legend(loc="upper left")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "scatter_gt_vs_facemap.png"), dpi=150)
# plt.show()
#
# # (c) GT vs Meye
# plt.figure(figsize=(5, 5))
# plt.scatter(gt_zero, det_Meye_zero, alpha=0.3, s=10, color="orange",
#             label=f"Meye (r={r_oth:.3f})")
# plt.axline((0,0), slope=1, color="k", linestyle="--", linewidth=0.7)
# plt.xlabel("GT (zero‐mean)")
# plt.ylabel("Meye pipeline (zero‐mean)")
# plt.title("Scatter: GT vs Meye")
# plt.legend(loc="upper left")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "scatter_gt_vs_Meye.png"), dpi=150)
# plt.show()
#
#
# # === 11) BLAND–ALTMAN PLOTS ========================================
# # (a) FaceIt
# diff_fi = det_faceit_zero - gt_zero
# mean_fi = (det_faceit_zero + gt_zero) / 2.0
# bias_fi = np.mean(diff_fi)
# loa_fiu = bias_fi + 1.96 * np.std(diff_fi)
# loa_fil = bias_fi - 1.96 * np.std(diff_fi)
#
# plt.figure(figsize=(6, 4))
# plt.scatter(mean_fi, diff_fi, alpha=0.3, s=10, color="red")
# plt.axhline(bias_fi, color="red", linestyle="-",  label=f"Bias={bias_fi:.2f}")
# plt.axhline(loa_fiu, color="gray", linestyle="--", linewidth=0.8, label=f"+1.96σ={loa_fiu:.2f}")
# plt.axhline(loa_fil, color="gray", linestyle="--", linewidth=0.8, label=f"-1.96σ={loa_fil:.2f}")
# plt.xlabel("Mean of GT & FaceIt")
# plt.ylabel("FaceIt – GT")
# plt.title("Bland–Altman: FaceIt vs GT")
# plt.legend(loc="upper right", fontsize="small")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "bland_altman_faceit.png"), dpi=150)
# plt.show()
#
# # (b) Facemap
# diff_fm = det_facemap_zero - gt_zero
# mean_fm = (det_facemap_zero + gt_zero) / 2.0
# bias_fm = np.mean(diff_fm)
# loa_fmu = bias_fm + 1.96 * np.std(diff_fm)
# loa_fml = bias_fm - 1.96 * np.std(diff_fm)
#
# plt.figure(figsize=(6, 4))
# plt.scatter(mean_fm, diff_fm, alpha=0.3, s=10, color="green")
# plt.axhline(bias_fm, color="red", linestyle="-",  label=f"Bias={bias_fm:.2f}")
# plt.axhline(loa_fmu, color="gray", linestyle="--", linewidth=0.8, label=f"+1.96σ={loa_fmu:.2f}")
# plt.axhline(loa_fml, color="gray", linestyle="--", linewidth=0.8, label=f"-1.96σ={loa_fml:.2f}")
# plt.xlabel("Mean of GT & Facemap")
# plt.ylabel("Facemap – GT")
# plt.title("Bland–Altman: Facemap vs GT")
# plt.legend(loc="upper right", fontsize="small")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "bland_altman_facemap.png"), dpi=150)
# plt.show()
#
# # (c) Meye
# diff_oth = det_Meye_zero - gt_zero
# mean_oth = (det_Meye_zero + gt_zero) / 2.0
# bias_oth = np.mean(diff_oth)
# loa_othu = bias_oth + 1.96 * np.std(diff_oth)
# loa_othl = bias_oth - 1.96 * np.std(diff_oth)
#
# plt.figure(figsize=(6, 4))
# plt.scatter(mean_oth, diff_oth, alpha=0.3, s=10, color="orange")
# plt.axhline(bias_oth, color="red", linestyle="-",  label=f"Bias={bias_oth:.2f}")
# plt.axhline(loa_othu, color="gray", linestyle="--", linewidth=0.8, label=f"+1.96σ={loa_othu:.2f}")
# plt.axhline(loa_othl, color="gray", linestyle="--", linewidth=0.8, label=f"-1.96σ={loa_othl:.2f}")
# plt.xlabel("Mean of GT & Meye")
# plt.ylabel("Meye – GT")
# plt.title("Bland–Altman: Meye vs GT")
# plt.legend(loc="upper right", fontsize="small")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(out_folder, "bland_altman_Meye.png"), dpi=150)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import cv2
image = np.load(r"C:\Users\faezeh.rabbani\ASSEMBLE\15-53-26\debug_face\1710514480.552814.npy", allow_pickle=True)
faceit_path = r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\faceit.npz"
faceit = np.load(faceit_path)
print(list(faceit.keys()))
"[top,bottom, left,right]"
faceit_frame_pos = faceit['frame_pos']
faceit_frame_axes = faceit['frame_axes']
faceit_frame_center = faceit['frame_center']
sub_region = image[ faceit_frame_pos[0][0]: faceit_frame_pos[0][1],faceit_frame_pos[0][2] :faceit_frame_pos[0][3]]
height, width = sub_region.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)
center = (width // 2, height // 2)
axes = (width // 2, height // 2)
cv2.ellipse(mask, center=center, axes=axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
if sub_region.ndim == 2:
    masked_processed = cv2.bitwise_and(sub_region, sub_region, mask=mask)
elif sub_region.ndim == 3 and sub_region.shape[2] == 3:
    masked_processed = cv2.bitwise_and(sub_region, sub_region, mask=mask)
elif sub_region.ndim == 3 and sub_region.shape[2] == 4:
    channels = cv2.split(sub_region)
    for i in range(3):
        channels[i] = cv2.bitwise_and(channels[i], channels[i], mask=mask)
    masked_processed = cv2.merge(channels)
check_pos = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\frame_pos.npy", allow_pickle=True)
print("check_pos", check_pos)
plt.imshow(masked_processed)
plt.show()
