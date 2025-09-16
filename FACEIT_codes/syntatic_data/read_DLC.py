import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Input file ----
file_path = Path(r"C:\Users\faezeh.rabbani\Downloads\1\SimulatedEyeVideos\SimulatedEyeVideos\Condition_8\5\EyeTracking_Data.h5")

# ---- Read H5 (DLC) ----
with h5py.File(file_path, "r") as f:
    w = f["w"]
    items  = [b.decode("utf-8") for b in w["block0_items"][:]]
    values = w["block0_values"][:]

df = pd.DataFrame(values, columns=items)
df.index.name = "frame"

# ---- Pick the series we need ----
req_cols = {
    "Pupil_Area": "Pupil_Area",
    "X_center": "Spot_Cent_x",
    "Y_center": "Spot_Cent_y",
}
missing = [v for v in req_cols.values() if v not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns in H5: {missing}\nAvailable: {list(df.columns)}")

summary = pd.DataFrame({
    "Pupil_Area (pixels^2)": df[req_cols["Pupil_Area"]],
    "X_center (pixels)":     df[req_cols["X_center"]],
    "Y_center (pixels)":     df[req_cols["Y_center"]],
}, index=df.index)

# ---- Plot & save image ----
fig = plt.figure(figsize=(11, 7))

plt.subplot(3, 1, 1)
plt.plot(summary["Pupil_Area (pixels^2)"], label="Pupil Area", color =  "coral")
plt.ylabel("Area (pixels²)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(summary["X_center (pixels)"], label="Pupil X Center", color =  "plum")
plt.ylabel("X Center (pixels)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(summary["Y_center (pixels)"], label="Pupil Y Center")
plt.xlabel("Frame")
plt.ylabel("Y Center (pixels)")
plt.legend()

plt.tight_layout()

# Output paths (same folder as the H5)
out_png  = file_path.with_name(file_path.stem + "_pupil_plots.png")
out_xlsx = file_path.with_name(file_path.stem + "_pupil_data.xlsx")

plt.savefig(out_png, dpi=150)
plt.close(fig)

# ---- Save Excel (summary + full table) ----
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="PupilSummary")
    df.to_excel(writer, sheet_name="AllColumns")


print(f"Saved image to: {out_png}")
print(f"Saved Excel to: {out_xlsx}")
