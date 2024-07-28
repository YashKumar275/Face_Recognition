import os 
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

torch.manual_seed(random.randint(1,1000))
## Modified U-Net (G_edit)
class ImageDataset(Dataset):
    def __init__(self,root_gt,root_masked,root_binary,transform=None):
        self.transform = transform
        self.files_gt = sorted(glob.glob(root_gt+"/*"+"/*.*"))
        self.files_masked = sorted(glob.glob(root_masked+"/*"+"/*.*"))
        self.files_binary = sorted(glob.glob(root_binary+"/*"+"/*.*"))
        
    def __getitem__(self,index):
        item_gt = self.transform(Image.open(self.files_gt[index%len(self.files_gt)]))
        item_masked = self.transform(Image.open(self.files_masked[index%len(self.files_masked)]))
        item_binary = self.transform(Image.open(self.files_binary[index%len(self.files_binary)]).convert("RGB"))
        return (item_gt-0.5)*2,(item_masked-0.5)*2,(item_binary-0.5)*2
    
    def __len__(self):
        return min(len(self.files_masked),len(self.files_binary))
    
def crop(image,new_shape):
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2
    starting_height = middle_height-round(new_shape[2]/2)
    final_height = starting_height+new_shape[2]
    starting_width = middle_width-round(new_shape[3]/2)
    final_width = starting_width+new_shape[3]
    cropped_image = image[:,:,starting_height:final_height,starting_width:final_width]
    return cropped_image
class ContractingBlock(nn.Module):
    def __init__(self,input_channels,use_in=True,use_dropout=False):
        super(ContractingBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels*2)
        self.use_in = use_in
        if use_dropout:
            self.drop = nn.Dropout()
        self.use_dropout = use_dropout
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

    
class ExpandingBlock(nn.Module):
    def __init__(self,input_channels,use_in=True):
        super(ExpandingBlock,self).__init__()
        self.tconv = nn.ConvTranspose2d(input_channels,input_channels//2,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv2 = nn.Conv2d(input_channels,input_channels//2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels//2)
        self.use_in = use_in
        
    def forward(self,x,skip_x):
        x = self.tconv(x)
        skip_x = crop(skip_x,x.shape)
        x = torch.cat([x,skip_x],axis=1)
        x = self.conv2(x)
        if self.use_in:
            x = self.insnorm(x)
        x = self.activation(x)
        return x
    
    
class FeatureMapBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(FeatureMapBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.conv(x)
        return x
    
    
class SE_Block(nn.Module):
    def __init__(self,channels,reduction=16):
        super(SE_Block,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b,c)
        y = self.excitation(y).view(b,c,1,1)
        return x * y.expand_as(x)
    
    
class AtrousConv(nn.Module):
    def __init__(self,input_channels):
        super(AtrousConv,self).__init__()
        self.aconv2 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=2,padding=2)
        self.aconv4 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=4,padding=4)
        self.aconv8 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=8,padding=8)
        self.aconv16 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=16,padding=16)
        self.batchnorm = nn.BatchNorm2d(input_channels)
        self.activation = nn.ReLU()
        
    def forward(self,x):
        x = self.aconv2(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv4(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv8(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv16(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        return x
    
class UNet(nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels=32):
        super(UNet,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False,use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels*2,use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels*4,use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.contract5 = ContractingBlock(hidden_channels*16)
        
        self.atrous_conv = AtrousConv(hidden_channels*32)
        
        self.expand0 = ExpandingBlock(hidden_channels*32)
        self.expand1 = ExpandingBlock(hidden_channels*16)
        self.expand2 = ExpandingBlock(hidden_channels*8)
        self.expand3 = ExpandingBlock(hidden_channels*4)
        self.expand4 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels,output_channels)
        
        self.se1 = SE_Block(hidden_channels*2)
        self.se2 = SE_Block(hidden_channels*4)
        self.se3 = SE_Block(hidden_channels*8)
        
        self.tanh = torch.nn.Tanh()
        
        
    def forward(self,x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x1 = self.se1(x1)
        x2 = self.contract2(x1)
        x2 = self.se2(x2)
        x3 = self.contract3(x2)
        x3 = self.se3(x3)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x5 = self.atrous_conv(x5)
        x6 = self.expand0(x5,x4)
        x7 = self.expand1(x6,x3)
        x8 = self.expand2(x7,x2)
        x9 = self.expand3(x8,x1)
        x10 = self.expand4(x9,x0)
        xn = self.downfeature(x10)
        
        return self.tanh(xn)
## Discriminator (whole region)
class Discriminator_whole(nn.Module):
    def __init__(self,input_channels,hidden_channels=8):
        super(Discriminator_whole,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1)  
        
    def forward(self,x,y):
        x = torch.cat([x,y],axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
## Discriminator (mask region)
class Discriminator_mask(nn.Module):
    def __init__(self,input_channels,hidden_channels=8):
        super(Discriminator_mask,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1) 
        self.dropout = nn.Dropout()
        
    def forward(self,x,y): 
        x = torch.cat([x,y],axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x2 = self.dropout(x2)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
### Parameters
import torch.nn.functional as F
adv_criterion = nn.BCEWithLogitsLoss()
#adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()
lambda_recon = 100
lambda_Dwhole = 0.3
lambda_Dmask = 0.7
lambda_adv_whole = 0.3
lambda_adv_mask = 0.7

n_epochs=1
input_dim = 6
output_dim = 3
disc_dim = 9
display_step = 879     # 10548img/3batch / 4
batch_size = 3
lr = 0.0003
target_shape = 224
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor()
])

gt_root = "unmasked_dataset2"      ### change ###
masked_root = "masked_dataset2"    ### change ###
binary_root = "binary_dataset2"    ### change ###
dataset = ImageDataset(gt_root,masked_root,binary_root,transform=transform)
