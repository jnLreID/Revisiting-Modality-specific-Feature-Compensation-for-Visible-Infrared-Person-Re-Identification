import numpy as np
from PIL import Image, ImageChops
from torchvision import transforms
import random
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import Transform as transforms
from model.s1model import Generatorfeature
from PIL import Image
import cv2

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
       
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        # print(self.train_color_label.shape,self.train_thermal_label.shape)
        # print(self.train_color_label.shape)
        
        # print(img1.shape,target1)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class SYSUData1(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Grayscale(num_output_channels=3)
                                        
        self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            # transforms.Pad(10),
                                            transforms.RectScale(288,144),
                                            transforms.RandomCrop((288,144)),
                                            transforms.RandomHorizontalFlip(),
                                        ])
        self.transform3 = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.gray = transforms.Grayscale(num_output_channels=3)
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = self.transform2(img1)
        img2 = self.transform2(img2)
        img3 = self.transform1(img1)
        img4 = self.transform1(img2)
        img1 = self.transform3(img1)
        img2 = self.transform3(img2)
        img3 = self.transform3(img3)
        img4 = self.transform3(img4)
        return img1, img2,img3, img4,target1, target2

    def __len__(self):
        return len(self.train_color_label)

        
class SYSUData2(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Grayscale(num_output_channels=3)
                                        
        self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            # transforms.Pad(10),
                                            transforms.RectScale(288,144),
                                            transforms.RandomCrop((288,144)),
                                            transforms.RandomHorizontalFlip(),
                                        ])
        self.transform3 = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.gray = transforms.Grayscale(num_output_channels=3)
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = self.transform1(img1)
        img2 = self.transform1(img2)
        img1 = self.transform2(img1)
        img2 = self.transform2(img2)
        img1 = self.transform3(img1)
        img2 = self.transform3(img2)
        img3 = self.transform3(img1)
        img4 = self.transform3(img2)

        # img3 = self.transform3(img3)
        # img4 = self.transform3(img4)
        return img1, img2,img3, img4,target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # RGB format
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class RegDBData1(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.train_color_label   = train_color_label
        self.train_thermal_label = train_thermal_label
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Grayscale(num_output_channels=3)
                                        
        self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            # transforms.Pad(10),
                                            transforms.RectScale(288,144),
                                            transforms.RandomCrop((288,144)),
                                            transforms.RandomHorizontalFlip(),
                                        ])
        self.transform3 = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.gray = transforms.Grayscale(num_output_channels=3)
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = self.transform2(img1)
        img2 = self.transform2(img2)
        img3 = self.transform1(img1)
        img4 = self.transform1(img2)
        img1 = self.transform3(img1)
        img2 = self.transform3(img2)
        img3 = self.transform3(img3)
        img4 = self.transform3(img4)

        return img1, img2,img3, img4,target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

    
    
class TestData1(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Grayscale(num_output_channels=3)
                                        
        self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            # transforms.Pad(10),
                                            transforms.RectScale(288,144),
                                            transforms.RandomCrop((288,144)),
                                            transforms.RandomHorizontalFlip(),
                                        ])
        self.transform3 = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform2(img1)
        img2 = self.transform1(img1)
        img1 = self.transform3(img1)
        img2 = self.transform3(img2)
        return img1, img2,target1
        
    def __len__(self):
        return len(self.test_image)

        
class TestData2(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Grayscale(num_output_channels=3)
                                        
        self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            # transforms.Pad(10),
                                            transforms.RectScale(288,144),
                                            transforms.RandomCrop((288,144)),
                                            transforms.RandomHorizontalFlip(),
                                        ])
        self.transform3 = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform2(img1)
        img2 = self.transform1(img1)
        img1 = self.transform3(img1)
        img2 = self.transform3(img2)
        return img1, img2,target1
        
        
    def __len__(self):
        return len(self.test_image)
        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

    
    
    