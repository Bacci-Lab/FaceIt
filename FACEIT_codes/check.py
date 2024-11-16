from pynwb import NWBHDF5IO
import numpy as np

data = np.load(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images/faceit.npz')
pupil_center = data['pupil_center']
motion_energy = data['motion_energy']
print(motion_energy)
# with NWBHDF5IO(r'C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\test_data\test_images/faceit.nwb', 'r') as io:
#     nwbfile = io.read()
#     processing = nwbfile.processing['eye facial movement']
#     pupil_center_data = processing.data_interfaces['pupil_center'].data[:]
#     print(pupil_center_data)