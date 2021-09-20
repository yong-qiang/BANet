import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .colorize_mask import cityscapes_colorize_mask
#import .edge_utils as edge_utils
#from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt


@HEADS.register_module()
class DAHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(DAHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        # convs = []
        # convs.append(
        #     ConvModule(
        #         self.in_channels,
        #         self.channels,
        #         kernel_size=kernel_size,
        #         padding=kernel_size // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg))
        # for i in range(num_convs - 1):
        #     convs.append(
        #         ConvModule(
        #             self.channels,
        #             self.channels,
        #             kernel_size=kernel_size,
        #             padding=kernel_size // 2,
        #             conv_cfg=self.conv_cfg,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))
        # if num_convs == 0:
        #     self.convs = nn.Identity()
        # else:
        #     self.convs = nn.Sequential(*convs)
        # if self.concat_input:
        #     self.conv_cat = ConvModule(
        #         self.in_channels + self.channels,
        #         self.channels,
        #         kernel_size=kernel_size,
        #         padding=kernel_size // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
###########################################2021.qy

    def forward(self,res):
        """Forward function."""
        output = res[5]                       
        return output 