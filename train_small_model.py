import numpy as np
import progressbar
import torch
import glob
import sys
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import deque
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from model.i3d import InceptionI3d, load_i3d_imagenet_pretrained
from model.classifier_layer import classifier_i3d
from torch.utils.tensorboard import SummaryWriter
from model.confusion_matrix_plotter import plot_conf_mat
from torchsummary import summary

# Last Layer of Inception 3D network
I3D_ENDPOINT = 'Mixed_4d'

# Last layer number of channels
OUT_CHANNElS = 512

if __name__ == '__main__':

    # Flow or RGB Training
    FLOW_TRAINING = True

    # Train the whole network or just the last layer
    WHOLE_NETWORK_TRAIN = True

    # Resume the training or generate new weights
    RESUME_TRAINING = True

    # Normalized Accuracy
    NORM_ACCURACY = False

    

    if FLOW_TRAINING:
        from dataloader_flow import action_dataset

        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.1
        # Temporal Sampling of the scenes
        T = 64
        if WHOLE_NETWORK_TRAIN:
            LR = 0.02
            checkpoint_save_path = 'model/smaller/flow_checkpoint_whole.pt'
            checkpoint_load_path_0 = 'model/smaller/flow_checkpoint_whole.pt'
            checkpoint_load_path_1 = 'model/smaller/flow_checkpoint.pt'
        else:
            checkpoint_save_path = 'model/smaller/flow_checkpoint.pt'
            checkpoint_load_path_1 = 'model/smaller/flow_checkpoint_whole.pt'
            checkpoint_load_path_0 = 'model/smaller/flow_checkpoint.pt'
    else:
        from dataloader import action_dataset

        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.2
        # Temporal Sampling of the scenes
        T = 64
        if WHOLE_NETWORK_TRAIN:
            LR = 0.02
            checkpoint_save_path = 'model/smaller/rgb_checkpoint_whole.pt'
            checkpoint_load_path_0 = 'model/smaller/rgb_checkpoint_whole.pt'
            checkpoint_load_path_1 = 'model/smaller/rgb_checkpoint.pt'
        else:
            checkpoint_save_path = 'model/smaller/rgb_checkpoint.pt'
            checkpoint_load_path_1 = 'model/smaller/rgb_checkpoint_whole.pt'
            checkpoint_load_path_0 = 'model/smaller/rgb_checkpoint.pt'
    print('Saving model on: ', checkpoint_save_path)
    # Number of scenes loaded into memory, 2 for whole network training, and 32 for last layer training
    # is recommended for 8Gb of VRAM
    if WHOLE_NETWORK_TRAIN:
        BATCH_SIZE = 6
    else:
        BATCH_SIZE = 14

    # Set up Logger on Tensorboard
    exp_name = 'small_exp_' + str(len(glob.glob('runs/**/*.0', recursive=True)))
    prefix = 'runs/'
    pre_trained = 'whole_network/' if WHOLE_NETWORK_TRAIN else 'last_layer/'
    writer_name_flow = prefix + 'flow/' + pre_trained + exp_name if FLOW_TRAINING else prefix + 'rgb/' + pre_trained + exp_name
    writer = SummaryWriter(writer_name_flow)

    # Device used for training
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')

    # Delta error for stagnation early stopping
    delta_error = 1e-2

    # Loads the database being used
    database = action_dataset(T, evaluate=False, small=True)

    # Splits the database between two sets, validation and training set.
    train_set, validation_set = torch.utils.data.random_split(database,
                                                              [len(database) - int(len(database) * 0.2),
                                                               int(len(database) * 0.2)])
    dataloader_train = DataLoader(train_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=6)

    dataloader_validation = DataLoader(validation_set,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=6)

    # Loads the inception network
    if FLOW_TRAINING:
        model = InceptionI3d(spatial_squeeze=False, in_channels=3, final_endpoint=I3D_ENDPOINT).to(device=DEVICE)
        ckp = torch.load("model/checkpoint.pt", map_location=DEVICE)
        model.load_state_dict(ckp, strict=False)
    else:
        model = InceptionI3d(spatial_squeeze=False, in_channels=3, final_endpoint=I3D_ENDPOINT).to(device=DEVICE)
        ckp = torch.load("model/checkpoint.pt", map_location=DEVICE)
        model.load_state_dict(ckp, strict=False)
    model.train()


    # Loads the last layer network, used to classify the actions
    model_last_layer = classifier_i3d(num_classes=6, in_channels=OUT_CHANNElS).to(device=DEVICE)

    # Loads the optimizer as Stochastic Gradient Descent and the Learning rate scheduler
    if WHOLE_NETWORK_TRAIN:
        optimizer = optim.Adam([
                                {'params': model_last_layer.parameters(), 'lr': LR},
                                {'params': model.parameters(), 'lr': LR}, 
                                ], 
                                lr=LR)
    else:
        optimizer = optim.SGD(model_last_layer.parameters(), 
                            lr=LR, momentum=0.9, weight_decay=1e-7)

    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=2, gamma=0.98)

    # Loss is Binary Cross Entropy with Logits
    if NORM_ACCURACY:
        loss = nn.CrossEntropyLoss(weight=database.pos_weight.to(DEVICE))
    else:
        loss = nn.CrossEntropyLoss()

    early_stop_loss = deque(maxlen=20)
    init_epoch = 0
    
    # Load last checkpoint
    if RESUME_TRAINING:
        
        # Try loading the first option of saved model - Same Type (Flow or RGB) and same training config (Whole network
        # or last layer)
        try:
            checkpoint = torch.load(checkpoint_load_path_0)
            model_last_layer.load_state_dict(checkpoint['last_layer_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            init_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_dict'])
            print('Loaded saved model: ', checkpoint_load_path_0)
        except:
            
            # If not found, try the second option of saved model - Same Type (Flow or RGB) and different training config
            # (Whole network or last layer)
            try:
                checkpoint = torch.load(checkpoint_load_path_1)
                model_last_layer.load_state_dict(checkpoint['last_layer_dict'])
                
                # Not loading the scheduler, optimizer and the epoch, as the training is getting only a similar network
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
                # init_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_dict'])
                print('Loading closer trained model: ', checkpoint_load_path_1)
            except:
                print('Error: ', sys.exc_info()[0])
                print('New model generated')

    for epoch in range(EPOCHS):
        epoch = epoch + init_epoch

        database.evaluate = False
        # Training Pass
        training_loss_list = []
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        i = 0
        for data in progressbar.progressbar(dataloader_train):
            i += 1
            model.train()
            inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)

            labels = torch.argmax(data['integer class'].to(device=DEVICE), dim=1)
            if WHOLE_NETWORK_TRAIN:
                output_i3d = model(inputs)
            else:
                with torch.no_grad():
                    output_i3d = model(inputs)

            output_last_layer = model_last_layer(output_i3d.detach())
            values = torch.mean(output_last_layer, dim=2)

            # Classification loss
            classification_loss = loss(values, labels)

            # Localization loss
            localization_loss = loss(torch.max(output_last_layer, dim=2)[0], labels)

            error = classification_loss/2 + localization_loss/2
            # error = classification_loss
            error.backward()

            # gradient accumulation
            if i % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            training_loss_list.append(error.detach().cpu().numpy())
        scheduler.step()

        # Logging
        training_loss = np.mean(training_loss_list)
        writer.add_scalar('Training Loss', training_loss, epoch)

        # Evaluation Pass
        database.evaluate = True
        validation_loss_list = []
        pred_name = []
        target = []
        weight = []
        for data in progressbar.progressbar(dataloader_validation):
            model.eval()
            with torch.no_grad():
                inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
                labels = torch.argmax(data['integer class'].to(device=DEVICE).float(), dim=1)
                output_i3d = model(inputs)
                output_last_layer = model_last_layer(output_i3d.detach())
                values = torch.mean(output_last_layer, dim=2)
                
                # Classification loss
                classification_loss = loss(values, labels)

                # Localization loss
                localization_loss = loss(torch.max(output_last_layer, dim=2)[0], labels)

                error = classification_loss/2 + localization_loss/2
                # error = classification_loss

                validation_loss_list.append(error.detach().cpu().numpy())

                pred, indices = torch.max(torch.log_softmax(values/2 + torch.max(output_last_layer, dim=2)[0]/2, dim=1), dim=1)
                for i_pred in indices:
                    pred_name.append(database.classes[i_pred])
                for i_label in torch.argmax(data['integer class'], dim=1):
                    weight.append(database.pos_weight[i_label].cpu().numpy())
                target_i = data['class']
                target += target_i
        database.evaluate = False
        current_loss = np.mean(validation_loss_list)

        # Logging
        weight = np.asarray(weight)
        target = np.asarray(target)
        pred_name = np.asarray(pred_name)

        # Logging with normalized accuracy or not
        if NORM_ACCURACY:
            accuracy = balanced_accuracy_score(target, pred_name, sample_weight=weight)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database.classes, weights=weight)
        else:
            accuracy = accuracy_score(target, pred_name)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database.classes)
            
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

        checkpoint = {'epoch': epoch,
                      'last_layer_dict': model_last_layer.state_dict(),
                      'model_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_save_path)
