import progressbar
import torch
from torch.utils.data import DataLoader
from torch import nn
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.model_importer import model_importer
from utils.evaluate import evaluate, conf_mat
from utils.confusion_matrix_plotter import plot_conf_mat
from utils.dataloader_rgb_flow import action_dataset

if __name__ == '__main__':

    # The device currently being used for evaluating
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')

    # Run the database with transforms or raw
    DATA_RAW = False

    # Run the test with visualization
    VISUALIZATION = False

    # Models to be used in evaluation
    # models = ['i3d-flow', 'i3d'] # Eval 1
    # models = ['s3d', 's3d-flow'] # Eval 2
    # models = ['i3d-shufflenet', 'i3d-shufflenet-flow'] # Eval 3
    # models = ['i3d-flow', 'i3d', 's3d', 's3d-flow'] # Eval 4
    # models = ['i3d-flow', 'i3d', 'i3d-shufflenet', 'i3d-shufflenet-flow'] # Eval 5
    models = ['i3d-flow', 'i3d', 'i3d-shufflenet', 'i3d-shufflenet-flow', 's3d', 's3d-flow'] # Eval 6
    # models = ['i3d-shufflenet-flow']


    torch_models = model_importer(models, torch.device('cuda:0'))
    
    print(torch_models.keys())

    # Batch Size is the number of sequences loaded into memory
    BATCH_SIZE = 8

    # T is the temporal size of the sequence of images.
    T = 64

    # The database for evaluation
    database = action_dataset(T, end_paths=['/test'])

    # Dataloader initialization
    dataloader = DataLoader(database,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)
    

    for model in torch_models:
        pytorch_total_params = sum(p.numel() for p in torch_models[model]['model'].parameters())
        print('Total Number of parameters: '+model, pytorch_total_params)

    pred_name = []
    target = []
    for data in progressbar.progressbar(dataloader):
        with torch.no_grad():
            # Get the inputs from the dataloader, permuting to keep Batch X Temporal Sequence X Channels X H X W
            inputs = [
                data['rgb video'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE), 
                data['flow video'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)]

            # Get the integer labels for the class in the format [0, 0, 0, 0, 1, 0] for example.
            labels = data['integer class'].to(device=DEVICE).float()
            class_name = data['class']

            # Get the real frames of the image
            real_frames = data['real frame']


            name, targ = evaluate(torch_models, inputs, class_name, database, real_frames, VISUALIZATION)
            pred_name += name
            target += targ
    print('-------------------------------- Evaluation --------------------------------')
    print('Used models: ', models)
    conf_mat(pred_name, target, database)
            

