import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_conf_mat(target, pred, labels, weights=None, Agg=True, big=False):
    if Agg:
        matplotlib.use('Agg')
    """
    Args:
        target:
        pred:
        labels:
    """
    if weights is not None:
        conf_mat = confusion_matrix(target, pred, labels=labels, sample_weight=weights)
    else:
        conf_mat = confusion_matrix(target, pred, labels=labels)
    size = (15, 15) if big else None
    fig, ax = plt.subplots(figsize=size)
    cax = ax.matshow(conf_mat)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)).tolist())
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)).tolist())
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    return fig