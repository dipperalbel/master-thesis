import numpy as np
import progressbar
import torch
import glob
import sys
import argparse

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import deque
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from model.s3d.s3d import S3D

from torch.utils.tensorboard import SummaryWriter
from utils.confusion_matrix_plotter import plot_conf_mat
from torchsummary import summary


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Custom parameters to run on a terminal machine')

    my_parser.add_argument('-batch_size',
                            action='store',
                            dest='batch_size',
                            type=int,
                            help='The size of the training batch',
                            default=1)
    
    my_parser.add_argument('-T',
                            action='store',
                            dest='T',
                            type=int,
                            help='The size of the temporal capture',
                            default=64)

    my_parser.add_argument('--resume_training',
                        action='store_true',
                        dest='resume_training',
                        help='Resume training or start from scratch',
                        default=False)

    my_parser.add_argument('--flow',
                        action='store_true',
                        dest='flow',
                        help='Setup for flow training',
                        default=False)

    args = my_parser.parse_args()

    # Number of scenes loaded into memory, 2 for whole network training, and 32 for last layer training
    # is recommended for 8Gb of VRAM
    BATCH_SIZE = args.batch_size
    print(f'Running With Batch Size: {BATCH_SIZE:d}')

    # Resume the training or generate new weights
    RESUME_TRAINING = args.resume_training
    print('Resuming training...' if RESUME_TRAINING else 'Starting new training...')

    
    # Flow or RGB Training
    FLOW_TRAINING = args.flow
    print('Training with flow dataset...' if FLOW_TRAINING else 'Training with RGB dataset...')

    # Temporal Sampling of the scenes
    T = args.T
    print(f'Training with temporal size T: {T:d}')

    # Normalized Accuracy
    NORM_ACCURACY = False

    

    if FLOW_TRAINING:
        from utils.dataloader_flow import action_dataset

        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.01


        LR = 0.01
        checkpoint_save_path = 'model/s3d/flow_checkpoint_whole.pt'

    else:
        from utils.dataloader import action_dataset

        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.01
        # Temporal Sampling of the scenes
        T = args.T

        checkpoint_save_path = 'model/s3d/rgb_checkpoint_whole.pt'
    
    print('Saving model on: ', checkpoint_save_path)

    ORIGINAL_BATCH_SIZE = 64
    GRADIENT_ACCUM_STEPS =  ORIGINAL_BATCH_SIZE // BATCH_SIZE

    # Set up Logger on Tensorboard
    exp_name = 's3d_exp_' + str(len(glob.glob('runs/**/*.0', recursive=True)))
    prefix = 'runs/'
    writer_name_flow = prefix + 'flow/' + exp_name if FLOW_TRAINING else prefix + 'rgb/' + exp_name
    writer = SummaryWriter(writer_name_flow)

    # Device used for training
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')

    # Delta error for stagnation early stopping
    delta_error = 1e-2

    # Loads the database being used
    database_train = action_dataset(
                            T=T,
                            T_eval=64,
                            end_paths=['/training'], 
                            small=False)
    
    database_eval = action_dataset(
                            T=T,
                            T_eval=64,
                            end_paths=['/test'], 
                            small=False)
    
    dataloader_train = DataLoader(database_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    dataloader_validation = DataLoader(database_eval,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=4)

    # Loads the inception network
    if FLOW_TRAINING:
        model = nn.DataParallel(S3D(num_class=6).to(device=DEVICE))
    else:
        model = nn.DataParallel(S3D(num_class=6).to(device=DEVICE))
    model.train()

    # Loads the optimizer as Stochastic Gradient Descent and the Learning rate scheduler
    optimizer = optim.SGD([
                            {'params': model.parameters(), 'lr': LR},
                            ], 
                            lr=LR, momentum=0.9, weight_decay=0.0000001)


    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=3, gamma=0.98)

    # Loss is Binary Cross Entropy with Logits
    if NORM_ACCURACY:
        loss = nn.BCEWithLogitsLoss(weight=database_train.pos_weight.to(DEVICE))
    else:
        loss = nn.BCEWithLogitsLoss()

    early_stop_loss = deque(maxlen=20)
    init_epoch = 0
    
    # Load last checkpoint
    if RESUME_TRAINING:        
        checkpoint = torch.load(checkpoint_save_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        print('Loaded saved model: ', checkpoint_save_path)
        

    for epoch in range(EPOCHS):
        epoch = epoch + init_epoch

        # Training Pass
        training_loss_list = []
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        i = 0
        for data in progressbar.progressbar(dataloader_train):
            i += 1
            model.train()
            inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)

            labels = data['integer class'].to(device=DEVICE)
            values = model(inputs)

            # Classification loss
            classification_loss = loss(values, labels)

            error = classification_loss
            error.backward()

            # gradient accumulation
            if i % GRADIENT_ACCUM_STEPS == 0 or i == len(dataloader_train):
                optimizer.step()
                optimizer.zero_grad()
            training_loss_list.append(error.detach().cpu().numpy())
        scheduler.step()

        # Logging
        training_loss = np.mean(training_loss_list)
        writer.add_scalar('Training Loss', training_loss, epoch)

        # Evaluation Pass
        validation_loss_list = []
        pred_name = []
        target = []
        weight = []
        for data in progressbar.progressbar(dataloader_validation):
            model.eval()
            with torch.no_grad():
                inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
                labels = data['integer class'].to(device=DEVICE)
                values = model(inputs)
                
                # Classification loss
                classification_loss = loss(values, labels)

                error = classification_loss

                validation_loss_list.append(error.detach().cpu().numpy())

                pred, indices = torch.max(torch.log_softmax(values, dim=1), dim=1)
                for i_pred in indices:
                    pred_name.append(database_eval.classes[i_pred])
                for i_label in torch.argmax(data['integer class'], dim=1):
                    weight.append(database_eval.pos_weight[i_label].cpu().numpy())
                target_i = data['class']
                target += target_i

        current_loss = np.mean(validation_loss_list)

        # Logging
        weight = np.asarray(weight)
        target = np.asarray(target)
        pred_name = np.asarray(pred_name)

        # Logging with normalized accuracy or not
        if NORM_ACCURACY:
            accuracy = balanced_accuracy_score(target, pred_name, sample_weight=weight)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_eval.classes, weights=weight)
        else:
            accuracy = accuracy_score(target, pred_name)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_eval.classes)
            
        acc_percentage = accuracy * 100
        writer.add_scalar('Accuracy', acc_percentage, epoch)
        writer.add_figure('Confusion Matrix', conf_mat_fig, epoch)
        validation_loss = current_loss
        writer.add_scalar('Validation Loss', validation_loss, epoch)
        writer.flush()

        
        # Early stopping technique -- Stagnation or rising validation loss
        # If the last three loss are 10% bigger than the moving average of loss, break
        early_stop_loss.append(current_loss)
        high_validation_loss = np.mean(np.array(early_stop_loss)[-3:]) > np.mean(early_stop_loss) * 1.1 and len(early_stop_loss) > 10
        
        
        # If the validation loss is stagnated, break
        stagnation_validation_loss = abs(np.mean(np.array(early_stop_loss)[-3:]) - np.mean(early_stop_loss)) / np.mean(
            early_stop_loss) < delta_error and len(early_stop_loss) > 10

        # if high_validation_loss or stagnation_validation_loss:
        #     print('Rising error on validation, early stop')
        #     break

        save_checkpoint = {
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        torch.save(save_checkpoint, checkpoint_save_path)
