from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

npz_path = Path(r"C:\Users\zbook\FaceIt\test_data\test_images\faceIt\faceit.npz")
with np.load(npz_path, allow_pickle=True) as z:
    print("Keys:", list(z.files))
    print("len Keys:", len(list(z.files)))
    Face_frame  = z["Face_frame"].copy()
    Pupil_frame = z["Pupil_frame"].copy()
    Pupil_dilation = z["pupil_dilation"].copy()
    motion_energy = z["motion_energy"].copy()
print(motion_energy)

print("Face_frame shape:", Face_frame)
print("Pupil_frame shape:", Pupil_frame)

plt.plot(motion_energy, linestyle="--")
plt.show()
plt.plot(Pupil_dilation)
plt.show()