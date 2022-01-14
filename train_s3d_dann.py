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
from model.s3d.s3d_dann import S3D
from model.domain_classifier import DomainClassifier
from utils.data_vis import data_visualization

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

    my_parser.add_argument('-dl_weight',
                        action='store',
                        dest='domain_loss_weight',
                        type=float,
                        help='Weight for the domain loss',
                        default=1.0)
        
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
    
    my_parser.add_argument('--source_only',
                    action='store_true',
                    dest='source',
                    help='Training with source only',
                    default=False)

    my_parser.add_argument('--hmdb',
                action='store_true',
                dest='hmdb',
                help='Training with hmdb as source',
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

    # HMDB Source or Target
    HMDB_Source = args.hmdb
    UFC_101_Source = False if HMDB_Source else True
    target_dataset = 'hmdb-51' if UFC_101_Source else 'ucf-101'
    source_dataset = 'ucf-101' if UFC_101_Source else 'hmdb-51'
    print(f'Taget dataset {target_dataset} \t Source dataset {source_dataset}')
    
    source_only = args.source
    print('Training with source only' if source_only else 'Training with DANN')
    
    source_str = 'source_only_' if source_only else ''
    flow_str = 'flow' if FLOW_TRAINING else 'rgb'
    path_name = 's3d_dann_' + source_str + source_dataset + '_' + flow_str + '.pt'
    
    domain_loss_weight = args.domain_loss_weight
    
    if FLOW_TRAINING:
        from utils.dataloader_dann_flow import action_dataset

        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.001

        checkpoint_save_path = 'model/s3d/'+path_name

    else:
        from utils.dataloader_dann import action_dataset
        
        # Number of epochs to run
        EPOCHS = 400
        # Learning Rate
        LR = 0.001

        checkpoint_save_path = 'model/s3d/'+path_name
    
    print('Saving model on: ', checkpoint_save_path)

    ORIGINAL_BATCH_SIZE = 32
    GRADIENT_ACCUM_STEPS =  ORIGINAL_BATCH_SIZE // BATCH_SIZE

    # Set up Logger on Tensorboard
    exp_name = 'i3d_exp_' + str(len(glob.glob('runs/**/*.0', recursive=True)))
    prefix = 'runs/'
    writer_name_flow = prefix + 'dann_flow/' + exp_name if FLOW_TRAINING else prefix + 'dann_rgb/' + exp_name
    writer = SummaryWriter(writer_name_flow)

    # Device used for training
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')

    # Delta error for stagnation early stopping
    delta_error = 1e-2

    # Loads the database being used
    database_train_target = action_dataset(
                                T=T,
                                end_paths=['/training'], 
                                dataset=target_dataset)
    
    database_train_source = action_dataset(
                            T=T,
                            end_paths=['/training'], 
                            dataset=source_dataset)
    
    database_eval = action_dataset(
                            T=T,
                            end_paths=['/test'],
                            dataset=target_dataset)
    
    database_eval_source = action_dataset(
                            T=T,
                            end_paths=['/test'],
                            dataset=source_dataset)
    
    
    dataloader_source = DataLoader(database_train_source,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)
    
    dataloader_target = DataLoader(database_train_target,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)

    dataloader_validation = DataLoader(database_eval,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4,
                                    drop_last=True)

    dataloader_validation_source = DataLoader(database_eval_source,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4,
                                    drop_last=True)

    # Loads the inception network
    if FLOW_TRAINING:
        model = nn.DataParallel(S3D(num_class=12).to(device=DEVICE))
        domain_classifier = nn.DataParallel(DomainClassifier(in_dim=7168, hidden_dim=64).to(device=DEVICE))
    else:
        model = nn.DataParallel(S3D(num_class=12).to(device=DEVICE))
        domain_classifier = nn.DataParallel(DomainClassifier(in_dim=7168, hidden_dim=64).to(device=DEVICE))
    model.train()

    # Loads the optimizer as Stochastic Gradient Descent and the Learning rate scheduler
    optimizer = optim.Adam([
                            {'params': model.parameters(), 'lr': LR},
                            {'params': domain_classifier.parameters(), 'lr': LR},
                            ], 
                            lr=LR)


    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=3, gamma=0.98)

    # Loss is Binary Cross Entropy with Logits
    if NORM_ACCURACY:
        classification_loss = nn.BCEWithLogitsLoss(weight=database_train_source.pos_weight.to(DEVICE))
        domain_loss = nn.BCEWithLogitsLoss()
    else:
        classification_loss = nn.BCEWithLogitsLoss()
        domain_loss = nn.BCEWithLogitsLoss()

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
        classification_loss_list = []
        source_domain_loss_list = []
        target_domain_loss_list = []
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        i = 0
        
        data_length = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in progressbar.progressbar(range(data_length)):
           
            alpha = 0.5
            
            model.train()
            
            # Source domain
            data_source = data_source_iter.next()
            # data_visualization(data_source, data_frame='source')
            inputs = data_source['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
            image_labels = data_source['integer class'].to(device=DEVICE)
            domain_labels = torch.tile(torch.tensor([1.0, 0.0]), (image_labels.shape[0], 1)).to(device=DEVICE)

            class_output, features = model(inputs)
            domain_output = domain_classifier(features)
            source_classification_loss = classification_loss(class_output, image_labels)
            source_domain_loss = domain_loss(domain_output, domain_labels)
            
            if source_only:
                loss = source_classification_loss
                source_domain_loss = torch.tensor(0.0)
                target_domain_loss = torch.tensor(0.0)
                
            else:
            
                # Target domain
                data_target = data_target_iter.next()
                # data_visualization(data_target, data_frame='target')
                inputs = data_target['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
                domain_labels = torch.tile(torch.tensor([0.0, 1.0]), (image_labels.shape[0], 1)).to(device=DEVICE)
                
                
                _, features = model(inputs)
                domain_output = domain_classifier(features)
                target_domain_loss = domain_loss(domain_output, domain_labels)
                
                # Classification loss
                loss = target_domain_loss + source_domain_loss + source_classification_loss

            loss.backward()

            # gradient accumulation
            if i % GRADIENT_ACCUM_STEPS == 0 or i == data_length - 1:
                optimizer.step()
                optimizer.zero_grad()
            training_loss_list.append(loss.detach().cpu().numpy())
            classification_loss_list.append(source_classification_loss.detach().cpu().numpy())
            source_domain_loss_list.append(source_domain_loss.detach().cpu().numpy())
            target_domain_loss_list.append(target_domain_loss.detach().cpu().numpy())
        scheduler.step()

        # Logging
        training_loss = np.mean(training_loss_list)
        classification_loss_log = np.mean(classification_loss_list)
        source_domain_loss = np.mean(source_domain_loss_list)
        target_domain_loss = np.mean(target_domain_loss_list)
        
        writer.add_scalar('Training Loss', training_loss, epoch)
        writer.add_scalar('Classification Loss', classification_loss_log, epoch)
        writer.add_scalar('Source/Domain Loss', source_domain_loss, epoch)
        writer.add_scalar('Target/Domain Loss', target_domain_loss, epoch)

        # Evaluation Pass Target
        validation_loss_list = []
        pred_name = []
        target = []
        weight = []
        for data in progressbar.progressbar(dataloader_validation):
            model.eval()
            with torch.no_grad():
                inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
                labels = data['integer class'].to(device=DEVICE)
                values, _ = model(inputs)
                
                # Classification loss
                error = classification_loss(values, labels)

                validation_loss_list.append(error.detach().cpu().numpy())

                pred, indices = torch.max(torch.log_softmax(values, dim=1), dim=1)
                for i_pred in indices:
                    pred_name.append(database_eval.converted_classes[i_pred])
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
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_train_source.converted_classes, weights=weight, big=True)
        else:
            accuracy = accuracy_score(target, pred_name)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_train_source.converted_classes, big=True)
        
        acc_percentage = accuracy * 100
        writer.add_scalar('Target/Accuracy', acc_percentage, epoch)
        
        writer.add_figure('Target/Confusion Matrix', conf_mat_fig, epoch)
        
        validation_loss = current_loss
        writer.add_scalar('Target/Validation Loss', validation_loss, epoch)
            
            
        # Evaluation Pass Source
        validation_loss_list = []
        pred_name = []
        target = []
        weight = []
        for data in progressbar.progressbar(dataloader_validation_source):
            model.eval()
            with torch.no_grad():
                inputs = data['video images'].permute(0, 4, 1, 2, 3).float().to(device=DEVICE)
                labels = data['integer class'].to(device=DEVICE)
                values, _ = model(inputs)
                
                # Classification loss
                error = classification_loss(values, labels)

                validation_loss_list.append(error.detach().cpu().numpy())

                pred, indices = torch.max(torch.log_softmax(values, dim=1), dim=1)
                for i_pred in indices:
                    pred_name.append(database_eval.converted_classes[i_pred])
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
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_train_source.converted_classes, weights=weight, big=True)
        else:
            accuracy = accuracy_score(target, pred_name)
            conf_mat_fig = plot_conf_mat(target=target, pred=pred_name, labels=database_train_source.converted_classes, big=True)
            
        acc_percentage = accuracy * 100
        writer.add_scalar('Source/Accuracy', acc_percentage, epoch)
        
        writer.add_figure('Source/Confusion Matrix', conf_mat_fig, epoch)
        
        validation_loss = current_loss
        writer.add_scalar('Source/Validation Loss', validation_loss, epoch)
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
        
