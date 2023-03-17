import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# own functions 
import functions

class UNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(UNet, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        # UNet network architecture
        self.conv1 = functions.double_conv(in_features, hidden_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = functions.double_conv(hidden_dim, hidden_dim*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = functions.double_conv(hidden_dim*2, hidden_dim*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = functions.double_conv(hidden_dim*4, hidden_dim*8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = functions.double_conv(hidden_dim*8, hidden_dim*16)
        self.up6 = nn.ConvTranspose2d(hidden_dim*16, hidden_dim*8, 2, stride=2)
        self.conv6 = functions.double_conv(hidden_dim*16, hidden_dim*8)
        self.up7 = nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, 2, stride=2)
        self.conv7 = functions.double_conv(hidden_dim*8, hidden_dim*4)
        self.up8 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 2, stride=2)
        self.conv8 = functions.double_conv(hidden_dim*4, hidden_dim*2)
        self.up9 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 2, stride=2)
        self.conv9 = functions.double_conv(hidden_dim*2, hidden_dim)
        self.conv10 = nn.Conv2d(hidden_dim, out_features, 1)
        
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = F.sigmoid(c10)
        return out        
