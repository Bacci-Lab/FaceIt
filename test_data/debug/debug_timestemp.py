from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



npz_path2 = Path(r"D:\faceIt\faceit.npz")
with np.load(npz_path2, allow_pickle=True) as z:
    print("Keys:", list(z.files))
    print("len Keys:", len(list(z.files)))
    Pupil_dilation_before = z["pupil_dilation"].copy()




#plt.figure(figsize=(8, 4))
plt.plot(Pupil_dilation_before)
plt.title("before debugging")

plt.show()