###################################################################################################
#ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1906.09826.pdf
###################################################################################################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import cv2
import numpy as np
#from torchsummary import summary
from .segbase import SegBaseModel     

class ASPP(nn.Module):
    def __init__(self, in_channel=2048, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth*5 , in_channel, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class ASPP1(nn.Module):


    def __init__(self, in_dim, reduction_dim=16, output_stride=8, rates=[6, 12, 18]):
        super(ASPP1, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)


        # 1x1
        self.features0 = nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim, eps=1e-3), nn.ReLU(inplace=True))
        # other rates

        self.features1 = nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=12, padding=12, bias=False),
                nn.BatchNorm2d(reduction_dim, eps=1e-3),
                nn.ReLU(inplace=True)
            )
        self.features2 = nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=24, padding=24, bias=False),
                nn.BatchNorm2d(reduction_dim, eps=1e-3),
                nn.ReLU(inplace=True)
            )       
        self.features3 = nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=36, padding=36, bias=False),
                nn.BatchNorm2d(reduction_dim, eps=1e-3),
                nn.ReLU(inplace=True)
            ) 
#        self.features4 = nn.Sequential(nn.Conv2d(80, in_dim, kernel_size=1, bias=False),
#                         nn.BatchNorm2d(in_dim, eps=1e-3), nn.ReLU(inplace=True))
        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, eps=1e-3), nn.ReLU(inplace=True))
            
        # edge level features
        self.edge_pooling = nn.AdaptiveAvgPool2d(1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, eps=1e-3), nn.ReLU(inplace=True))
        
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim+64,in_dim,(1,1),bias=False),nn.BatchNorm2d(in_dim,eps=1e-3),nn.ReLU(inplace=True))


    def forward(self, x, edge):
        x_size = x.size()
        if x.shape !=edge.shape:
            edge=nn.Upsample(size=x.shape[-2:], scale_factor=None, mode='bilinear', align_corners=True)(edge)        
        seg_edge1 = torch.cat((x,edge),1)
        #seg_edge1=x+edge
#seg
        seg = self.img_pooling(x)
        seg = self.img_conv(seg)
        seg = F.interpolate(seg, x_size[2:],
                                     mode='bilinear',align_corners=True)
#edge
#        edge = self.edge_pooling(edge)
#        edge = self.edge_conv(edge)
#        edge = F.interpolate(edge, x_size[2:],
#                                     mode='bilinear',align_corners=True)        
#ASPP
        seg_edge1 = self.conv1_1(seg_edge1)
        cat1 = self.features0(seg_edge1)
        cat2 = self.features1(seg_edge1)
        cat3 = self.features2(seg_edge1)
        cat4 = self.features3(seg_edge1)   
        cat = torch.cat((cat1,cat2,cat3,cat4),1)

        seg_out = torch.cat((seg,cat),1)             
#        edge_out = torch.cat((edge,cat),1)  
        return seg_out
        
class SAB(nn.Module):
    def __init__(self,inplanes,planes):
        super().__init__()
       
##compress channel
        self.conv1_1 = nn.Conv2d(inplanes,inplanes//16,(1,1),1,bias=True)  
        self.conv1_2 = nn.Conv2d(planes,inplanes//16,(1,1),1,bias=True)   
##sa             
        self.conv1x1_1 = nn.Conv2d(inplanes//16,1,(1,1),stride=1,bias=True)  
#        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
#        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid2 = nn.Sigmoid()      
#        self.sigmoid3 = nn.Sigmoid()         
#        self.sigmoid = nn.Sigmoid()
        self.conv3x3 = nn.Conv2d(inplanes//8,inplanes//16,(3,3),stride=1,padding=1,bias=True)

    def forward(self,x,y,h):
        h1,w1 = x.size()[2:]
        h2,w2 = y.size()[2:]
        if h1==h2 and w1==w2:
            y0 = y
            y0 = self.conv1_2(y0)
        else:
            y0 = F.interpolate(y, size=(h1, w1), mode="bilinear", align_corners=True)
            y0 = self.conv1_2(y0)       
##SA     
        x0 = self.conv1_1(x)   
        x1 = self.conv1x1_1(x0)
#        x1_h = self.pool1(x1)
#        x1_w = self.pool2(x1)
        x1 = self.sigmoid2(x1)
#        x1_w = self.sigmoid3(x1_w)
##test
#        print(x.shape)
#        print(y0.shape)
#        print(z.shape)        
        out = torch.cat((x0,y0),1)
        out_c = self.conv3x3(out)
##add_weight
#        w = torch.cat((x1,h),1)
#        w1 = self.conv1_3(w)
        out = out_c*x1
        front_ = out
        out = (out - out.min()) / (out.max() - out.min())
        front = out
        diffY = h.size()[2] - out.size()[2]
        diffX = h.size()[3] - out.size()[3]

        out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out1 = out + h
        later = out1
        diffY = out1.size()[2] - out_c.size()[2]
        diffX = out1.size()[3] - out_c.size()[3]

        out_c = F.pad(out_c, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out2 = out1 + out_c
        
        return F.relu(out2),front_,front,later



###originl GAB
class AEB(nn.Module):
    def __init__(self,inplanes,planes):
        super().__init__()
                
#        self.up = UpsamplerBlock(planes,inplanes)
        self.conv = nn.Conv2d(planes,inplanes,(1,1),1,bias=True)
##CA        
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(inplanes+planes,inplanes,(1,1),stride=1,bias=True)
        self.sigmoid1 = nn.Sigmoid()
##New SA
        self.conv1x1_2 = nn.Conv2d(inplanes,1,(1,1),stride=1,bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid2 = nn.Sigmoid()      
        self.sigmoid3 = nn.Sigmoid()           
##SA        
#        self.conv1x1_2 = nn.Conv2d(inplanes,1,(1,1),stride=1,bias=True)
#        self.sigmoid2 = nn.Sigmoid()
               
        self.conv1x1_3 = nn.Conv2d(planes, inplanes,(1,1),1,bias=True)
              
    def forward(self,x,y):
        h1,w1 = x.size()[2:]
        h2,w2 = y.size()[2:]

        if h1==h2 and w1==w2:
            y0 = y
            y1 = self.conv1x1_3(y)
#            print(y1.shape)
        else:
            y0 = y
            y1 = F.interpolate(y, size=(h1, w1), mode="bilinear", align_corners=True)    
            y1 = self.conv(y1)     
        x1 = x
##SA        
        x2 = self.conv1x1_2(x1)
        x2_h = self.pool1(x2)
        x2_w = self.pool2(x2)
        x2_h = self.sigmoid2(x2_h)
        x2_w = self.sigmoid3(x2_w)
##CA        
        x3 = self.avgpool1(x1)
        y2 = self.avgpool2(y0)
        
        y3 = torch.cat((x3,y2),1)
        y3 = self.conv1x1_1(y3)
        y3 = self.sigmoid1(y3)

        y1_ = x2_h*y1+x2_w*y1
        y1_1 = y3*y1_

        y1_out = y1_1+y1
        return F.relu(y1_out) 

class SBNetibn4(SegBaseModel):
    def __init__(self, nclass, backbone='resnet101', aux=False, pretrained_base=True, **kwargs):
        super(SBNetibn4,self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        
##downsampling
        self.conv3x3_1 = nn.Conv2d(4,4,(3,3),stride=2,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(4, eps=1e-3)
        self.conv3x3_2 = nn.Conv2d(4,4,(3,3),stride=2,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-3)
        self.conv3x3_3 = nn.Conv2d(4,4,(3,3),stride=2,padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
##sigmoid
        self.sigmoid = nn.Sigmoid()
##upsampling                
        self.conv = nn.Conv2d(80, nclass, (1,1), 1, bias=True)
        self.conv1 = nn.Conv2d(80,1,(1,1),1,bias=True)
##AEB        
        self.aeb0 = AEB(1024,2048)
        self.aeb1 = AEB(512,1024)
        self.aeb2 = AEB(256,512)
##SAB
        self.sab0 = SAB(1024,2048,4)
        self.sab1 = SAB(512,1024,4)
        self.sab2 = SAB(256,512,4)  
##aspp
        self.aspp = ASPP() 
        self.aspp1 = ASPP1(256,16,8)    
 
    def forward(self,input):
        x_size = input.size()

        h,w = input.size()[2:] 
##image gradient
        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        
        canny = torch.cat((input,canny),1)

#        print(canny.shape)
        canny1 = self.relu(self.bn2(self.conv3x3_2(self.relu(self.bn1(self.conv3x3_1(canny))))))
 #       print(canny1.shape)
        canny2 = self.relu(self.bn3(self.conv3x3_3(canny1)))
#        print(canny2.shape)
        output0, output1, output2, output3 = self.base_forward(input)
#        print(output0.shape)
        output3 = self.aspp(output3)
##seg_out
        seg_out1 = self.aeb0(output2,output3)
        seg_out2 = self.aeb1(output1,seg_out1)
#        print(output2.shape)
        seg_out3 = self.aeb2(output0,seg_out2)
        
##edge_out
        edge_out1 = self.sab0(output2,output3,canny2)
        edge_out2 = self.sab1(output1,edge_out1,canny2)
        edge_out3 = self.sab2(output0,edge_out2,canny1)



        seg_out,edge_out = self.aspp1(seg_out3,edge_out3)        
        edge_out = self.conv1(edge_out)

        
        seg_out = self.conv(seg_out)
        seg_out = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=True)


        edge_out = F.interpolate(edge_out, size=(h, w), mode="bilinear", align_corners=True) 
        edge_out = self.sigmoid(edge_out)
        
        return seg_out,edge_out

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SBNet(nclass=11).to(device)
   # summary(model,(3,360,480))
