import torch
import os
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.autograd import Variable
from model.s1model import Generatorfeature
import Transform as transforms

# os.makedirs("sample-rrft" , exist_ok=True)
# os.makedirs("sample-rtfr" , exist_ok=True)
# ############################################################
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(2048)
    def forward(self, x):
        module_input = x
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = x1 + x2
        x = self.sigmoid(x)

        return x*module_input
        
# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
            model_ft2 = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
                
        for mo in model_ft2.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.visible = model_ft
        self.visible2 = model_ft2
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.sqex1 = SEModule(channels=256, reduction=16)
        self.sqex2 = SEModule(channels=512, reduction=32)
        self.sqex3 = SEModule(channels=1024, reduction=32)
        self.sqex4 = SEModule(channels=2048, reduction=32)
    def forward(self, x,x2):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.sqex3(x)
        x = self.visible.layer4(x)
        x = self.sqex4(x)
        xo1 = x 
        
        x2 = self.visible2.conv1(x2)
        x2 = self.visible2.bn1(x2)
        x2 = self.visible2.relu(x2)
        x2 = self.visible2.maxpool(x2)
        x2 = self.visible2.layer1(x2)
        x2 = self.visible2.layer2(x2)
        x2 = self.visible2.layer3(x2)
        x = self.sqex3(x)
        x2 = self.visible2.layer4(x2)
        x = self.sqex4(x)
        xo2 = x2
        
        x = (xo1 + xo2)/2
        
        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))
        
        vv = xo1.size(2) / 2
        vv = int(vv)
        kxo1 = xo1.size(2) - vv * (2 - 1)
        kxo1 = int(kxo1)
        xo1 = nn.functional.avg_pool2d(xo1, kernel_size=(kxo1, xo1.size(3)), stride=(vv, xo1.size(3)))
        xo1 = xo1.view(xo1.size(0), xo1.size(1), xo1.size(2))
        
        tt = xo2.size(2) / 2
        tt = int(tt)
        kxo2 = xo2.size(2) - tt * (2 - 1)
        kxo2 = int(kxo2)
        xo2 = nn.functional.avg_pool2d(xo2, kernel_size=(kxo2, xo2.size(3)), stride=(tt, xo2.size(3)))
        xo2 = xo2.view(xo2.size(0), xo2.size(1), xo2.size(2))
        
        return x,xo1,xo2
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
            model_ft2 = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        for mo in model_ft2.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
                
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.thermal2 = model_ft2
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.sqex1 = SEModule(channels=256, reduction=16)
        self.sqex2 = SEModule(channels=512, reduction=32)
        self.sqex3 = SEModule(channels=1024, reduction=32)
        self.sqex4 = SEModule(channels=2048, reduction=32)
    def forward(self, x,x2):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        x = self.thermal.layer3(x)
        x = self.sqex3(x)
        x = self.thermal.layer4(x)
        x = self.sqex4(x)
        xo1 = x 
        
        x2 = self.thermal2.conv1(x2)
        x2 = self.thermal2.bn1(x2)
        x2 = self.thermal2.relu(x2)
        x2 = self.thermal2.maxpool(x2)
        x2 = self.thermal2.layer1(x2)
        x2 = self.thermal2.layer2(x2)
        x2 = self.thermal2.layer3(x2)
        x = self.sqex3(x)
        x2 = self.thermal2.layer4(x2)
        x = self.sqex4(x)
        xo2 = x2
        
        
        x = (xo1 + xo2)/2
        
        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2)) 
        
        vv = xo1.size(2) / 2
        vv = int(vv)
        kxo1 = xo1.size(2) - vv * (2 - 1)
        kxo1 = int(kxo1)
        xo1 = nn.functional.avg_pool2d(xo1, kernel_size=(kxo1, xo1.size(3)), stride=(vv, xo1.size(3)))
        xo1 = xo1.view(xo1.size(0), xo1.size(1), xo1.size(2))
        
        tt = xo2.size(2) / 2
        tt = int(tt)
        kxo2 = xo2.size(2) - tt * (2 - 1)
        kxo2 = int(kxo2)
        xo2 = nn.functional.avg_pool2d(xo2, kernel_size=(kxo2, xo2.size(3)), stride=(tt, xo2.size(3)))
        xo2 = xo2.view(xo2.size(0), xo2.size(1), xo2.size(2))
        return x,xo1,xo2
        
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        if arch =='resnet34':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 2048
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature7 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature8 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature9 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature10 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier7 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier8 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier9 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier10 = ClassBlock(low_dim, class_num, dropout = drop)
        self.l2norm = Normalize(2)
        self.fakeT = Generatorfeature()
        self.fakeR = Generatorfeature()
        self.fakeT.loadlayer('visible to fake infrared generate model.pth')
        self.fakeR.loadlayer('infrared to fake visible generate model.pth')
    def forward(self, x1, x2, x3, x4, modal = 0):
        x3 = self.fakeT(x3)
        x4 = self.fakeR(x4)
        x3 = x3.detach()
        x4 = x4.detach()
        # save_image(x1, "sample-rrft/random-rr.png" , normalize=True)
        # save_image(x2, "sample-rtfr/random-rt.png" , normalize=True)
        # save_image(x3, "sample-rrft/random-ft.png" , normalize=True)
        # save_image(x4, "sample-rtfr/random-fr.png" , normalize=True)
        if modal==0:
            x1,v,ft = self.visible_net(x1,x3)
            #real RGB fusion fake thermal
            x1 = x1.chunk(6,2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0),-1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            #real RGB 
            v = v.chunk(2,2)
            v1 = v[0].contiguous().view(v[0].size(0),-1)
            v2 = v[1].contiguous().view(v[1].size(0),-1)
            #fake thermal 
            ft = ft.chunk(2,2)
            ft1 = ft[0].contiguous().view(ft[0].size(0),-1)
            ft2 = ft[1].contiguous().view(ft[1].size(0),-1)
            x2,t,fv = self.thermal_net(x2,x4)
            #real thermal fusion fake RGB
            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            #real thermal
            t = t.chunk(2,2)
            t1 = t[0].contiguous().view(t[0].size(0),-1)
            t2 = t[1].contiguous().view(t[1].size(0),-1)
            #fake RGB
            fv = fv.chunk(2,2)
            fv1 = fv[0].contiguous().view(fv[0].size(0),-1)
            fv2 = fv[1].contiguous().view(fv[1].size(0),-1)
            #real Image fusion Fake image
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
            #Real RGB-Image fusion Real T-image
            vt_0 = torch.cat((v1, t1), 0)
            vt_1 = torch.cat((v2, t2), 0)
            #fake RGB-Image fusion fake T-image
            fvft_0 = torch.cat((fv1, ft1), 0)
            fvft_1 = torch.cat((fv2, ft2), 0)
            
            y_6 = self.feature7(vt_0)
            y_7 = self.feature8(vt_1)
            y_8 = self.feature9(fvft_0)
            y_9 = self.feature10(fvft_1)
            out_6 = self.classifier7(y_6)
            out_7 = self.classifier8(y_7)
            out_8 = self.classifier9(y_8)
            out_9 = self.classifier10(y_9)
            
        elif modal ==1:
            x,v,ft = self.visible_net(x1,x3)
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0),-1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal ==2:
            x,t,fv = self.thermal_net(x2,x4)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        #Paried real and fake image 
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        #Paried real and fake image 
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5, 
            out_6,out_7, out_8, out_9), (
            self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3),
            self.l2norm(y_4), self.l2norm(y_5), self.l2norm(y_6), self.l2norm(y_7),
            self.l2norm(y_8), self.l2norm(y_9))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            return x, y
            
            
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    net = GeneratorResNet()
    x = torch.rand(2, 3, 288, 144)
    out = net(x)

    