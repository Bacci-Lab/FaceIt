import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Inputs ---
csv_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_1\calibration_dilated.csv"
invert_y_for_image_coords = True

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Sanity check ---
cols_needed = ["frame", "center_x", "center_y"]
missing = [c for c in cols_needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df[cols_needed].dropna().sort_values("frame")
df["frame"] = df["frame"].astype(int)

# --- Plot 1: x and y vs frame ---
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["center_x"], label="center_x")
plt.plot(df["frame"], df["center_y"], label="center_y")
plt.xlabel("Frame")
plt.ylabel("Center (pixels)")
plt.title("Pupil Center vs. Frame")
plt.legend()
plt.tight_layout()

outdir = Path(csv_path).parent
plt.savefig(outdir / "center_vs_frame.png", dpi=200)
plt.show()

# --- Plot 2: 2D trajectory ---
plt.figure(figsize=(6, 6))
plt.plot(df["center_x"], df["center_y"], marker="o", markersize=2, linewidth=1)
plt.scatter(df["center_x"].iloc[0], df["center_y"].iloc[0], s=40, label="start")
plt.scatter(df["center_x"].iloc[-1], df["center_y"].iloc[-1], s=40, label="end")

plt.xlabel("center_x (px)")
plt.ylabel("center_y (px)")
plt.title("Pupil Center Trajectory")
plt.gca().set_aspect("equal", adjustable="box")
if invert_y_for_image_coords:
    plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()

plt.savefig(outdir / "center_xy_trajectory.png", dpi=200)
plt.show()
