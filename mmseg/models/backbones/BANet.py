import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import cv2
import numpy as np
#from torchsummary import summary
    
        
class SAB(nn.Module):
    def __init__(self,inplanes,planes):
        super().__init__()
       
##compress channel
        self.conv1_1 = nn.Conv2d(inplanes,inplanes//16,(1,1),1,bias=True)  
        self.conv1_2 = nn.Conv2d(planes,inplanes//16,(1,1),1,bias=True)   
##sa             
        self.conv1x1_1 = nn.Conv2d(inplanes//16,1,(1,1),stride=1,bias=True)  
        self.sigmoid2 = nn.Sigmoid()      
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
        x1 = self.sigmoid2(x1)
        
        out = torch.cat((x0,y0),1)
        out_c = self.conv3x3(out)
##add_weight
        out = out_c*x1
        out = (out - out.min()) / (out.max() - out.min())
        
        diffY = h.size()[2] - out.size()[2]
        diffX = h.size()[3] - out.size()[3]
        out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out1 = out + h

        diffY = out1.size()[2] - out_c.size()[2]
        diffX = out1.size()[3] - out_c.size()[3]
        out_c = F.pad(out_c, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out2 = out1 + out_c
        
        return F.relu(out2)



###originl GAB
class GAB(nn.Module):
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

        y1_1 = y3*y1
        y1_ = x2_h*y1_1+x2_w*y1_1


        y1_out = y1_1+y1
        return F.relu(y1_out) 
