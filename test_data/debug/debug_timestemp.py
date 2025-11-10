from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

npz_path = Path(r"D:\faceIt_UPDATE\faceit.npz")
with np.load(npz_path, allow_pickle=True) as z:
    print("Keys:", list(z.files))
    print("len Keys:", len(list(z.files)))
    Face_frame  = z["Face_frame"].copy()
    Pupil_frame = z["Pupil_frame"].copy()
    Pupil_dilation = z["pupil_dilation"].copy()
    motion_energy = z["motion_energy"].copy()


print("Face_frame shape:", Face_frame)
print("Pupil_frame shape:", Pupil_frame)


npz_path2 = Path(r"D:\faceIt\faceit.npz")
with np.load(npz_path2, allow_pickle=True) as z:
    print("Keys:", list(z.files))
    print("len Keys:", len(list(z.files)))
    Pupil_dilation_before = z["pupil_dilation"].copy()



plt.figure(figsize=(8, 4))
plt.plot(Pupil_dilation, color= "red" )
plt.title("after debugging")

#plt.figure(figsize=(8, 4))
plt.plot(Pupil_dilation_before)
plt.title("before debugging")

plt.show()