import torch
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from utils.confusion_matrix_plotter import plot_conf_mat
# The font used on the overlay of the scene
font = cv.FONT_HERSHEY_SIMPLEX

def screen_show(database, T, inputs, real_frames, indices, sorted_indexes):
    # Print the results and the scene on a window
    for scene, real_frame, i_class, sort in zip(inputs, real_frames, indices, sorted_indexes):
        for i_photo in range(T):
            argmax = i_class[0]
            scnd_argmax = i_class[1]
            class_str = database.classes[argmax] + ' {:.2%}'.format(sort[0])
            scnd_class_str = database.classes[scnd_argmax] + ' {:.2%}'.format(sort[1])
            image = scene[:, i_photo, :, :].permute(1, 2, 0).detach().cpu().numpy()


            frame = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            frame_real_i = cv.normalize(real_frame[i_photo, :, :, :].cpu().numpy(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            image = cv.hconcat([frame, frame_real_i])

            cv.putText(image, class_str+' '+scnd_class_str, (10, 30), font, 0.4, (255, 255, 255), 1)
            cv.imshow('test', image)
            cv.waitKey(41)

def evaluate(model_dict, video_input, class_name, database, real_frames, visualization):
    T = database.T
    
    # Get the output of the network
    prediction = None
    for model in model_dict:
        if model_dict[model]['type'] == 'rgb':
            inputs = video_input[0]
        else:
            inputs = video_input[1]
        model_prediction = model_dict[model]['model'](inputs)
        if prediction is None:
            prediction = model_prediction
        else:
            prediction += model_prediction
    
    prediction = torch.log_softmax(prediction, dim=1)

    sorted_indexes, indices = torch.sort(prediction, dim=1, descending=True)

    if visualization:
        screen_show(database, T, inputs, real_frames, indices, sorted_indexes)
    
    pred_name = []   
    target = [] 
    pred, indices = torch.max(prediction, dim=1)
    for i_pred in indices:
        pred_name.append(database.classes[i_pred])
    target_i = class_name
    target += target_i

    return pred_name, target

def conf_mat(pred_name, target, database):
    _ = plot_conf_mat(target=target, pred=pred_name, labels=database.classes, Agg=False)
    
    correct_guess = 0
    for target_i, pred_name_i in zip(target, pred_name):
        correct_guess += 1 if target_i == pred_name_i else 0
    accuracy = correct_guess / len(target) * 100
    print(f'The accuracy was: {accuracy:.2f}')
    plt.show()