import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable
# from s2model import thermal_net_resnet, visible_net_resnet



class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat1, feat2, label1, label2):
		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, self.dist(center1, center2) - self.margin)
					# print(dist)
				else:
					dist += max(0, self.dist(center1, center2) - self.margin)
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, 1-self.dist(center1, center2) - self.margin)
		# print(i)
		# print(dist)
		# exit()
		return dist

class hetero_loss1(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss1, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat1, feat2, label1, label2):
		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		#loss = Variable(.cuda())
		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, self.dist(center1, center2) - self.margin)
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, 1-self.dist(center1, center2) - self.margin)
		return dist


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.3, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
		
'''
class kl_loss(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(kl_loss, self).__init__(size_average, reduce, reduction)
        self.target1 = Generatorweight()
        self.target1.loadlayer('/data/LJN/PyTorch-GAN-master/implementations/cyclegan/saved_model/sysu/15G_AB_30.pth')
        self.target2 = Generatorweight()
        self.target2.loadlayer('/data/LJN/PyTorch-GAN-master/implementations/cyclegan/saved_model/sysu/15G_BA_30.pth')
        self.input1 = visible_net_resnet()
        self.input2 = thermal_net_resnet()
    def forward(self, input = self.input1, target = self.target1):
        return F.kl_div(input, target, reduction=self.reduction)
'''