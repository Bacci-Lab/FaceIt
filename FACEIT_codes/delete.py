import numpy as np
data = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\faceit.npz", allow_pickle=True)
angle = data['angle']

frame_axes = data['frame_center']
print(frame_axes)

data2 = np.load(r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images\FaceIt\pupil_frame_pos.npy", allow_pickle=True)
print("data2", data2)