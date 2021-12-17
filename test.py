from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from data_loader import SYSUData1, RegDBData, TestData1
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from generatormodel import Generatorweight
from utils import *
import time 
import scipy.io as scio
import Transform as transforms

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
# parser.add_argument('--dataset', default='regdb',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model10.51/', type=str, help='model save path')
# parser.add_argument('--model_path', default='reg_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log2/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int, metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=24, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str, metavar='m', help='Method type')
parser.add_argument('--drop', default=0.0, type=float, metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=8, type=int, help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.0, type=float, help='weight of Hetero-Center Loss')
parser.add_argument('--thd', default=0, type=float, help='threshold of Hetero-Center Loss')
parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
random.seed(1)# 为所有GPU设置随机种子

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data/RGBT-reid/SYSU_MM011/'
    log_path = args.log_path + 'sysu_log/'
    n_class = 395
    test_mode = [1, 2] 
elif dataset =='regdb':
    data_path = '/data/RGBT-reid/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    n_class = 206
    test_mode = [2, 1]
    
if not os.path.isdir(log_path):
    os.makedirs(log_path)

if args.method =='id':
    suffix = dataset + '_id_bn_relu'
suffix = suffix + '_drop_{}'.format(args.drop)
suffix = suffix + '_lr_{:1.1e}'.format(args.lr)
suffix = suffix + '_dim_{}'.format(args.low_dim)
suffix = suffix + '_whc_{}'.format(args.w_hc)
suffix = suffix + '_thd_{}'.format(args.thd)
suffix = suffix + '_pimg_{}'.format(args.per_img)
suffix = suffix + '_gm_{}'.format(args.gall_mode)
suffix = suffix + '_m_{}'.format(args.mode)
test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch)
net.to(device)    
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        print('==> loading checkpoint {}'.format(args.resume), file=test_log_file)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']), file=test_log_file)
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
        print('==> no checkpoint found at {}'.format(args.resume), file=test_log_file)


if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
print('==> Loading data..', file=test_log_file)
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.RectScale(args.img_h, args.img_w),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0, gall_mode=args.gall_mode)

      
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    
    gallset  = TestData1(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
nquery = len(query_label)
ngall = len(gall_label)
print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

print("Dataset statistics:", file=test_log_file)
print("  ------------------------------", file=test_log_file)
print("  subset   | # ids | # images", file=test_log_file)
print("  ------------------------------", file=test_log_file)
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery), file=test_log_file)
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall), file=test_log_file)
print("  ------------------------------", file=test_log_file)

queryset = TestData1(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))   
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))
print('Data Loading Time:\t {:.3f}'.format(time.time()-end), file=test_log_file)

feature_dim = args.low_dim

if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    print('Extracting Gallery Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 6*feature_dim))
    gall_feat_pool = np.zeros((ngall, 6*pool_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            pool_feat, feat = net(input,input,input2,input2,test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            gall_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_pool
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    print('Extracting Query Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 6*feature_dim))
    query_feat_pool = np.zeros((nquery, 6*pool_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            pool_feat, feat = net(input,input,input2,input2,test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            query_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    return query_feat, query_feat_pool 
    
query_feat, query_feat_pool = extract_query_feat(query_loader)    

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_gall_feat(gall_loader)
    # fc feature 
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP  = eval_regdb(-distmat, query_label, gall_label)
    
    # pool5 feature
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

    print ('Test Trial: {}'.format(args.trial))
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))

    print('Test Trial: {}'.format(args.trial), file=test_log_file)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP), file=test_log_file)
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)
    
elif dataset =='sysu':
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = trial, gall_mode=args.gall_mode)
        
        trial_gallset = TestData1(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
        trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        # print(trial_gall_loader)
        # exit()
        gall_feat, gall_feat_pool = extract_gall_feat(trial_gall_loader)
        
        # fc feature 
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP  = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
        
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial ==0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        
        print ('Test Trial: {}'.format(trial))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        print('mAP: {:.2%}'.format(mAP_pool))

        print('Test Trial: {}'.format(trial), file=test_log_file)
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
        print('mAP: {:.2%}'.format(mAP), file=test_log_file)
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
        print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)

    cmc = all_cmc /10 
    mAP = all_mAP /10

    cmc_pool = all_cmc_pool /10 
    mAP_pool = all_mAP_pool /10
    print ('All Average:')
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))

    print('All Average:', file=test_log_file)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP), file=test_log_file)
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)