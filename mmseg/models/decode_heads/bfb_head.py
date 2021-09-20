import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule
from PIL import Image
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt
###seg to edge
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
#######
class Boundary_gt(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Boundary_gt, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, gtmasks):
        size = gtmasks.size()
        label = np.zeros((size[0], 1, size[2], size[3]))
        for i in range(size[0]):
          lab = gtmasks[i].cpu().detach().numpy()
          #print(label.shape)
          lab = lab.transpose(1,2,0)
          #print(label.shape)
          lab = lab.squeeze(2)     
###edge GT      
          lab = lab.copy()  
          mask = Image.fromarray(lab.astype(np.uint8))
#        _edgemap = mask.numpy()
          _edgemap = np.array(mask)
        
          _edgemap = label_to_onehot(_edgemap, 19)

          label[i] = onehot_to_binary_edges(_edgemap, 2, 19)

        label = torch.from_numpy(label).cuda().long()  
        #print(label.shape)
        return label
        #return boudary_targets_pyramid.long()


@HEADS.register_module()
class BFBHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 use_boundary = False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.use_boundary = use_boundary
        super(BFBHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        if self.use_boundary:
            self.get_boundary=Boundary_gt()


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        #print(x.shape)
        output = x
        #output = self.convs(x)
        #if self.concat_input:
            #output = self.conv_cat(torch.cat([x, output], dim=1))
        #output = self.cls_seg(output)
        return output