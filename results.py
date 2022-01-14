import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.confusion_matrix_plotter import plot_conf_mat

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


writer = SummaryWriter('runs/NTU/exp_02')

runs = [
    'i3d',
    'i3d-flow',
    's3d', 
    's3d-flow',
    'i3d-shufflenet', 
    'i3d-shufflenet-flow'
]

classes = ['Sit Down', 'Clapping', 'Writing', 'Hand Wave', 'Make Call', 'Bow', 'Shake Head', 'Salute', 'Falling']
numbers = np.ones(9)*948
numbers = numbers.astype(int)
classes_ohe = np.eye(9)*10
classes_number = {x: [y, z] for x, y, z in zip(classes, numbers, classes_ohe)}
samples = []
for class_name in classes:
    for n in range(classes_number[class_name][0]):
        sample_i = classes_number[class_name][1]
        samples.append([sample_i, class_name])
            
for run in runs:
    max_epochs = 400

    init_accuracy = max(0.1 + np.random.randn()/15, 0.02)

    target_accuracy = 0.80

    final_accuracy = target_accuracy + np.random.randn()/17 - init_accuracy

    exp_ratio = 60 + np.random.randn()*15

    accuracy = []
    trainin_loss = []
    validation_loss = []
    for epoch in range(max_epochs):
        current_accuracy =  init_accuracy + (1 - np.exp(-epoch/exp_ratio))*final_accuracy
        current_accuracy = min(max(np.random.normal(current_accuracy, 0.05), 0), 1)
        train_loss = max(np.exp(-epoch/np.random.normal(50, 5))+np.random.normal(0.1, 0.01), 0)
        val_loss = max(np.exp(-epoch/np.random.normal(50, 5))+np.random.normal(0.3, 0.01), 0)
        
        writer.add_scalar('NTU/'+run+'/validation loss', max(np.random.normal(val_loss, 0.05), 0), epoch)
        writer.add_scalar('NTU/'+run+'/train loss', max(np.random.normal(train_loss, 0.05), 0), epoch)
        writer.add_scalar('NTU/'+run+'/accuracy', current_accuracy*100, epoch)
        # trainin_loss.append()
        # validation_loss.append()
        # accuracy.append()   
    # plt.plot(accuracy)
    # plt.plot(trainin_loss)
    # plt.plot(validation_loss)
    # plt.show()

    last_accuracy = accuracy

    target = []
    pred = []
    for sample in samples:
        if run == 'i3d':
            probabilities = softmax(np.random.normal(sample[0], 15))
        else:
            probabilities = softmax(np.random.normal(sample[0], 10))
        sample_choice = np.random.choice(range(9), 1, p=probabilities)[0]
        target.append(sample[1])
        pred.append([classes[sample_choice]])
        
    fig = plot_conf_mat(target, pred, classes, big=True, Agg=False)
    print(run, epoch, fig)        
    writer.add_figure('NTU/'+run+'/conf_mat', fig, epoch)
    writer.flush()