"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

def label_to_onehot(label, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _label = [label == (i + 1) for i in range(num_classes)]
    return np.array(_label).astype(np.uint8)

def onehot_to_label(label):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _label = np.argmax(label, axis=0)
    _label[_label != 0] += 1
    return _label

def onehot_to_multiclass_edges(label, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return label
    
    # We need to pad the borders for boundary conditions
    label_pad = np.pad(label, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(label_pad[i, :])+distance_transform_edt(1.0-label_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)
        
    return np.array(channels)

def onehot_to_binary_edges(label, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """
    
    if radius < 0:
        return label
    
    # We need to pad the borders for boundary conditions
    label_pad = np.pad(label, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    edgemap = np.zeros(label.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(label_pad[i, :])+distance_transform_edt(1.0-label_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

