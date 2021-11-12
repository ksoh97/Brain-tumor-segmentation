import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tensorflow as tf
import tensorflow.keras.backend as K

# Keras
def DiceLoss(targets, inputs, smooth=1.0):
    # flatten label and prediction tensors
    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

# Keras
def DiceBCELoss(targets, inputs, smooth=1.0):
    # flatten label and prediction tensors
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

# Keras
def IoULoss(targets, inputs, smooth=1.0):
    # flatten label and prediction tensors
    inputs = np.where(inputs > 0.5, 1, inputs)
    inputs = np.where(inputs <= 0.5, 0, inputs)

    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def iou_score(targets, inputs, smooth=1.0):
    # flatten label and prediction tensors
    inputs = np.where(inputs > 0.5, 1, inputs)
    inputs = np.where(inputs <= 0.5, 0, inputs)

    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return IoU