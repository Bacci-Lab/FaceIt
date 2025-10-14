import numpy as np
data = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\faceit.npz", allow_pickle=True)
angle = data['angle']
print(angle)