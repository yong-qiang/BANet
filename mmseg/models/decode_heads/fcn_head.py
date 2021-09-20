import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import cv2
import numpy as np
from ..builder import HEADS
from .decode_head import BaseDecodeHead

i = 0
@HEADS.register_module()
class FCNHead(BaseDecodeHead):
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
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        # conv_padding = (kernel_size // 2) * dilation
        # convs = []
        # convs.append(
        #     ConvModule(
        #         self.in_channels,
        #         self.channels,
        #         kernel_size=kernel_size,
        #         padding=conv_padding,
        #         dilation=dilation,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg))
        # for i in range(num_convs - 1):
        #     convs.append(
        #         ConvModule(
        #             self.channels,
        #             self.channels,
        #             kernel_size=kernel_size,
        #             padding=conv_padding,
        #             dilation=dilation,
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
###qy
        #self.conv_seg = nn.Conv2d(256,19,(1,1),1,bias=True)
        #self.count = 1
    def forward(self, res):
        """Forward function."""
        #self.count += 1
        #test1 = res[0].cpu()
        #test1_ = test1.squeeze(0)             
        #test1_ = torch.mean(test1_,dim=0)  
        #test1_ = test1_.numpy()         
        #test1_ = (test1_ - np.min(test1_)) / (np.max(test1_) - np.min(test1_))
        #test1_ = np.uint8(255*test1_)
        #cv2.imwrite("/data1/qy3/qy2/boundaryexperiment/qy101/mmsegmentation/edges_visual/"+str(self.count)+'edge1.png',test1_)
        output = res[4]
        return output