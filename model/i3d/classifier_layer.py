import torch
import torch.nn as nn
from model.i3d.i3d import Unit3D, MaxPool3dSamePadding

class classifier_i3d(nn.Module):
    def __init__(self, num_classes, in_channels=1024):
        """
        Last layer for the i3d model, used to receive the last output from i3d and convert it to a classifier.
        Normally just training this layer is enough for a good prediction

        Args:
            num_classes: Number of classes to be classified
        """

        self.in_channels = in_channels
        super(classifier_i3d, self).__init__()
        self.logits = Unit3D(in_channels=in_channels,
                             output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='Classifier')

        

    def forward(self, x):
        """
        Args:
            x: input of the forward pass
        """        
        x = self.logits(x)
        logits = x.squeeze(3).squeeze(3)
        return logits