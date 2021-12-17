import torch.nn as nn
import torch.nn.functional as F
import torch
import os
# from unet_parts import *
from torchvision import models
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')


# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# cuda = torch.cuda.is_available()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places,stride=1, downsampling=False, expansion = 4 ):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling  
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(places*self.expansion),
            )
        if self.downsampling:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(places*self.expansion),
                )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsampling(x)
        out += residual
        out = self.relu(out)
        return out
##############################
#           RESNET
##############################
##############################
#        Generator
##############################

class GeneratorResNet(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(GeneratorResNet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        # avg pooling to global pooling
        #################Encoder###############
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = model_ft.conv1
        self.bn1 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        
        #################Decoder###############
        self.deconv1 = nn.Sequential(
        Bottleneck(2048, 512), 
        Bottleneck(2048, 256, downsampling=True),
       )
        self.deconv2 = nn.Sequential(
        Bottleneck(1024, 256), 
        Bottleneck(1024, 128, downsampling=True),
       )
        self.deconv3 = nn.Sequential(
        Bottleneck(512, 128), 
        Bottleneck(512, 64, downsampling=True),
       )        
        self.deconv4 = nn.Sequential(
        Bottleneck(256, 64), 
        Bottleneck(256, 32, downsampling=True),
       )
        self.deconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.up1 = upsample(1024, 1024)
        self.up2 = upsample(512, 512)
        self.up3 = upsample(256, 256)
        self.up4 = upsample(128, 64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)     
        x2 = self.layer1(x2)      ####256###
        x3 = self.layer2(x2)   ###512###
        x4 = self.layer3(x3)    ###1024###
        x5 = self.layer4(x4)    ###2048###
        dx1 = self.deconv1(x5)   ###1024###
        dx2 = dx1 + x4
        dx2 = self.deconv2(dx2)
        dx3 = self.up2(dx2) + x3 
        dx3 = self.deconv3(dx3)
        dx4 = self.up3(dx3) + x2
        dx4 = self.deconv4(dx4)
        dx5 = self.up4(dx4) + x1
        dx5 = self.deconv5(dx5)
        return dx5



'''
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 128)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, out_channels)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down4 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, out_channels)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        input_shape = self.outc(x)
        return input_shape
'''

class Generatorweight(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(Generatorweight, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        # avg pooling to global pooling
        #################Encoder###############
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = model_ft.conv1
        self.bn1 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)     
        x2 = self.layer1(x2)      ####256###
        x3 = self.layer2(x2)   ###512###
        x4 = self.layer3(x3)    ###1024###
        x5 = self.layer4(x4)    ###2048###
        return x5, x4, x3, x2, x1
    def loadlayer(self, path):
        if os.path.isfile(path):    
            pretrained_dict = torch.load(path)
            model_dict = self.state_dict()
            # 将pretrained_dict里不属于model_dict的键剔除掉
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            # 加载我们真正需要的state_dict
            self.load_state_dict(model_dict)
            print('weight success loaded')
        else: 
            print('file is not exist')


class Generatorfeature(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(Generatorfeature, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        # avg pooling to global pooling
        #################Encoder###############
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = model_ft.conv1    
        self.bn1 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        #################Decoder###############
        self.deconv1 = nn.Sequential(
        Bottleneck(2048, 512), 
        Bottleneck(2048, 256, downsampling=True),
       )
        self.deconv2 = nn.Sequential(
        Bottleneck(1024, 256), 
        Bottleneck(1024, 128, downsampling=True),
       )
        self.deconv3 = nn.Sequential(
        Bottleneck(512, 128), 
        Bottleneck(512, 64, downsampling=True),
       )        
        self.deconv4 = nn.Sequential(
        Bottleneck(256, 64), 
        Bottleneck(256, 32, downsampling=True),
       )
        self.deconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.up1 = upsample(1024, 1024)
        self.up2 = upsample(512, 512)
        self.up3 = upsample(256, 256)
        self.up4 = upsample(128, 64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)     
        x2 = self.layer1(x2)    ####256###
        x3 = self.layer2(x2)    ###512###
        x4 = self.layer3(x3)    ###1024###
        x5 = self.layer4(x4)    ###2048###
        dx1 = self.deconv1(x5)   ###1024###
        dx2 = dx1 + x4
        dx2 = self.deconv2(dx2)
        dx3 = self.up2(dx2) + x3 
        dx3 = self.deconv3(dx3)
        dx4 = self.up3(dx3) + x2
        dx4 = self.deconv4(dx4)
        dx5 = self.up4(dx4) + x1
        dx5 = self.deconv5(dx5)
        return dx5
    def loadlayer(self, path):
        if os.path.isfile(path):    
            pretrained_dict = torch.load(path)
            model_dict = self.state_dict()
            # 将pretrained_dict里不属于model_dict的键剔除掉
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            # 加载我们真正需要的state_dict
            self.load_state_dict(model_dict)
            print('weight success loaded')
        else: 
            print('file is not exist')


##############################
#        Discriminator       #
##############################
        
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 128, normalize=False),
            *discriminator_block(128, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

if __name__ == '__main__':
    net = GeneratorResNet()
    x = torch.rand(3, 3, 288, 144)
    out = net(x)
    # print(out.shape)
    #    num_residual_blocks = 9
#    G_AB = GeneratorUNet(input_shape, num_residual_blocks)
#    D_A = Discriminator(input_shape)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # in_channels = 3
    # out_channels = 3
    # net1 = Generatorweight().cuda()
    # net2 = Generatorweight().cuda()
    # net1.loadlayer('/G_vt.pth')
    # net2.loadlayer('/G_tv.pth')
    # net1 = GeneratorResNet().cuda()
    # net2 = GeneratorResNet().cuda()
    # x = torch.rand(2, 3, 288 , 144).cuda().float()
    # out1 = net1(x)
    # out2 = net2(x)
    # print(out1.shape)
    # print(out2.shape)
    