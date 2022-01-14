import cv2 as cv
import numpy as np

def data_visualization(list_of_data, data_frame):
    list_of_data = [list_of_data] if type(list_of_data) is dict else list_of_data
    ms_timestep = int(1/24 * 1000)
    for data in list_of_data:
        print(data['real frame'].shape, data['class'])
        for frame in data['real frame'][0]:
            frame = frame.cpu().numpy().astype(np.uint8)
            cv.imshow(data_frame, frame)
            cv.waitKey(ms_timestep)