from pathlib import Path
import numpy as np

npz_path = Path(r"C:\Users\zbook\FaceIt\test_data\test_images\faceIt\faceit.npz")
with np.load(npz_path, allow_pickle=True) as z:
    print("Keys:", list(z.files))
    Face_frame  = z["Face_frame"].copy()
    Pupil_frame = z["Pupil_frame"].copy()

print("Face_frame shape:", Face_frame)
print("Pupil_frame shape:", Pupil_frame)
