# -*- coding: UTF-8 -*-    
from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from data_loader import SYSUData1, RegDBData, TestData1
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from generatemodel import Generatorfeature
from utils import *
import Transform as transforms
from heterogeneity_loss import hetero_loss
import xlwt,xlrd
from torch.backends import cudnn
from setproctitle import setproctitle
setproctitle('python')
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# parser.add_argument('--dataset', default='regdb',  help='dataset name: regdb or sysu]')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only') 
parser.add_argument('--model_path', default='save_model10.51/', type=str, help='model save path')
parser.add_argument('--log_path', default='log10.51/', type=str,  help='log save path')
# parser.add_argument('--model_path', default='reg_model/', type=str, help='model save path')
# parser.add_argument('--log_path', default='reg/', type=str, help='log save path')
parser.add_argument('--save_epoch', default=1, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int, metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=24, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str, metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float, metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=6, type=int, help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.5, type=float, help='weight of Hetero-Center Loss')
parser.add_argument('--thd', default=0, type=float, help='threshold of Hetero-Center Loss')
parser.add_argument('--epochs', default=300, type=int, help='number of Training epoch')
parser.add_argument('--dist-type', default='l2', type=str, help='type of distance')
opt = parser.parse_args()
print(opt)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)   # 为所有GPU设置随机种子
np.random.seed(1)
random.seed(1)
#############################################RE-ID network##############################

def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(0 + worker_id)
    
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#fix_random(0)

    
dataset = args.dataset
if dataset == 'sysu':
    data_path = '/your data path/'
    log_path = args.log_path + 'sysu_log/'##记录log
    test_mode = [1, 2] ##################### thermal to visible
elif dataset =='regdb':
    data_path = '/your data path/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1] ##################### visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
    
if args.method =='id':
    suffix = 'model'
if dataset =='regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)
test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
feature_dim = args.low_dim

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RectScale(args.img_h, args.img_w),
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
    # training set
    trainset = SYSUData1(data_path,  transform=None)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')

gallset  = TestData1(gall_img, gall_label,transform = transform_test, img_size =(args.img_w,args.img_h))
queryset = TestData1(query_img, query_label,transform = transform_test, img_size =(args.img_w,args.img_h))
# testing data loader
gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init_fn)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init_fn)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)


print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')   
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))
print('==> Building model..')

net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch)
net.to(device)
# cudnn.benchmark = True
cudnn.benckmark = False
cudnn.deterministic = True

if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

if args.method =='id':
    thd = args.thd
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    criterion_het = hetero_loss(margin=thd, dist_type=args.dist_type)
    criterion_het.to(device)
ignored_params = list(map(id, net.feature1.parameters())) \
                 + list(map(id, net.feature2.parameters())) \
                 + list(map(id, net.feature3.parameters())) \
                 + list(map(id, net.feature4.parameters())) \
                 + list(map(id, net.feature5.parameters())) \
                 + list(map(id, net.feature6.parameters())) \
                 + list(map(id, net.feature7.parameters())) \
                 + list(map(id, net.feature8.parameters())) \
                 + list(map(id, net.feature9.parameters())) \
                 + list(map(id, net.feature10.parameters())) \
                 + list(map(id, net.classifier1.parameters())) \
                 + list(map(id, net.classifier2.parameters())) \
                 + list(map(id, net.classifier3.parameters()))\
                 + list(map(id, net.classifier4.parameters()))\
                 + list(map(id, net.classifier5.parameters()))\
                 + list(map(id, net.classifier6.parameters()))\
                 + list(map(id, net.classifier7.parameters()))\
                 + list(map(id, net.classifier8.parameters())) \
                 + list(map(id, net.classifier9.parameters()))\
                 + list(map(id, net.classifier10.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

if args.optim == 'sgd':
    optimizer = optim.SGD([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': net.feature1.parameters(), 'lr': args.lr},
         {'params': net.feature2.parameters(), 'lr': args.lr},
         {'params': net.feature3.parameters(), 'lr': args.lr},
         {'params': net.feature4.parameters(), 'lr': args.lr},
         {'params': net.feature5.parameters(), 'lr': args.lr},
         {'params': net.feature6.parameters(), 'lr': args.lr},
         {'params': net.feature7.parameters(), 'lr': args.lr},
         {'params': net.feature8.parameters(), 'lr': args.lr},
         {'params': net.feature9.parameters(), 'lr': args.lr},
         {'params': net.feature10.parameters(), 'lr': args.lr},
         {'params': net.classifier1.parameters(), 'lr': args.lr},
         {'params': net.classifier2.parameters(), 'lr': args.lr},
         {'params': net.classifier3.parameters(), 'lr': args.lr},
         {'params': net.classifier4.parameters(), 'lr': args.lr},
         {'params': net.classifier5.parameters(), 'lr': args.lr},
         {'params': net.classifier6.parameters(), 'lr': args.lr},
         {'params': net.classifier7.parameters(), 'lr': args.lr},
         {'params': net.classifier8.parameters(), 'lr': args.lr},
         {'params': net.classifier9.parameters(), 'lr': args.lr},
         {'params': net.classifier10.parameters(), 'lr': args.lr}],
         weight_decay=5e-4, momentum=0.9, nesterov=True)
elif args.optim == 'adam':
    optimizer = optim.Adam([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': net.feature.parameters(), 'lr': args.lr},
         {'params': net.classifier.parameters(), 'lr': args.lr}],weight_decay=5e-4)
         
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40 and epoch < 60:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
        
       
    optimizer.param_groups[0]['lr'] = 0.1*lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr
    optimizer.param_groups[5]['lr'] = lr
    optimizer.param_groups[6]['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr
    optimizer.param_groups[8]['lr'] = lr
    optimizer.param_groups[9]['lr'] = lr
    optimizer.param_groups[10]['lr'] = lr
    optimizer.param_groups[11]['lr'] = lr
    optimizer.param_groups[12]['lr'] = lr
    optimizer.param_groups[13]['lr'] = lr
    optimizer.param_groups[14]['lr'] = lr
    optimizer.param_groups[15]['lr'] = lr
    optimizer.param_groups[16]['lr'] = lr
    optimizer.param_groups[17]['lr'] = lr
    optimizer.param_groups[18]['lr'] = lr
    optimizer.param_groups[19]['lr'] = lr
    optimizer.param_groups[20]['lr'] = lr

    return lr

def train(epoch, loss_log):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()
    #num_batch = 0
    #epoch_loss = 0
    for batch_idx, (input1, input2, input3, input4,label1, label2) in enumerate(trainloader):
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        input3 = Variable(input3.cuda())
        input4 = Variable(input4.cuda())
        labels = torch.cat((label1,label2),0)
        labels = Variable(labels.cuda().long())
        label1 = Variable(label1.cuda().long())
        label2 = Variable(label2.cuda().long())
        data_time.update(time.time() - end)
        
        outputs, feat = net(input1, input2, input3, input4)
        if args.method =='id':
            loss0 = criterion(outputs[0], labels)
            loss1 = criterion(outputs[1], labels)
            loss2 = criterion(outputs[2], labels)
            loss3 = criterion(outputs[3], labels)
            loss4 = criterion(outputs[4], labels)
            loss5 = criterion(outputs[5], labels)
            loss6 = criterion(outputs[6], labels)
            loss7 = criterion(outputs[7], labels)
            loss8 = criterion(outputs[8], labels)
            loss9 = criterion(outputs[9], labels)

            het_feat0 = feat[0].chunk(2, 0)
            het_feat1 = feat[1].chunk(2, 0)
            het_feat2 = feat[2].chunk(2, 0)
            het_feat3 = feat[3].chunk(2, 0)
            het_feat4 = feat[4].chunk(2, 0)
            het_feat5 = feat[5].chunk(2, 0)
            het_feat6 = feat[6].chunk(2, 0)
            het_feat7 = feat[7].chunk(2, 0)
            het_feat8 = feat[8].chunk(2, 0)
            het_feat9 = feat[9].chunk(2, 0)

            loss_c0 = criterion_het(het_feat0[0], het_feat0[1], label1, label2)
            loss_c1 = criterion_het(het_feat1[0], het_feat1[1], label1, label2)
            loss_c2 = criterion_het(het_feat2[0], het_feat2[1], label1, label2)
            loss_c3 = criterion_het(het_feat3[0], het_feat3[1], label1, label2)
            loss_c4 = criterion_het(het_feat4[0], het_feat4[1], label1, label2)
            loss_c5 = criterion_het(het_feat5[0], het_feat5[1], label1, label2)
            loss_c6 = criterion_het(het_feat6[0], het_feat6[1], label1, label2)
            loss_c7 = criterion_het(het_feat7[0], het_feat7[1], label1, label2)
            loss_c8 = criterion_het(het_feat8[0], het_feat8[1], label1, label2)
            loss_c9 = criterion_het(het_feat9[0], het_feat9[1], label1, label2)

            loss0 = loss0 + 0.5 * loss_c0 
            loss1 = loss1 + 0.5 * loss_c1 
            loss2 = loss2 + 0.5 * loss_c2 
            loss3 = loss3 + 0.5 * loss_c3 
            loss4 = loss4 + 0.5 * loss_c4 
            loss5 = loss5 + 0.5 * loss_c5 
            loss6 = loss6 + 0.5 * loss_c6 
            loss7 = loss7 + 0.5 * loss_c7 
            loss8 = loss8 + 0.5 * loss_c8 
            loss9 = loss9 + 0.5 * loss_c9 

            _, predicted = outputs[0].max(1)
            correct += predicted.eq(labels).sum().item()
        optimizer.zero_grad()
        torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7,loss8, loss9],
        [torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(), 
        torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()])
        #loss.backward()
        optimizer.step()
        loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 6 + (loss6+ loss7 + loss8+ loss9)/ 4
        train_loss.update(loss.item(), 2*input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'Accu: {:.2f}' .format(
                  epoch, batch_idx, len(trainloader),current_lr, 
                  100.*correct/total, batch_time=batch_time, 
                  data_time=data_time, train_loss=train_loss))
        #num_batch = num_batch + 1
    #epoch_loss = epoch_loss / num_batch
    #loss_log.append(epoch_loss)


        
def test(epoch):   
    # switch to evaluation mode
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 6*args.low_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            feat_pool, feat = net(input,input,input2,input2,test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))

    # switch to evaluation mode
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 6*args.low_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            feat_pool, feat = net(input,input,input2,input2,test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    
    start = time.time()
    # compute the similarity
    distmat  = np.matmul(query_feat, np.transpose(gall_feat))
    
    # evaluation
    if dataset =='regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset =='sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time()-start))
    return cmc, mAP
    
# training
print('==> Start Training...')
per_img = args.per_img
per_id = args.batch_size / per_img
w_hc = args.w_hc
# w_tri = args.w_tri
loss_log = []
for epoch in range(start_epoch, args.epochs+1-start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
        trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size, per_img)
    trainset.cIndex = sampler.index1 # color index
    trainset.tIndex = sampler.index2 # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,\
        sampler = sampler, num_workers=args.workers, drop_last =True)
    
    # training 调用定义训练
    train(epoch, loss_log)

    if epoch >= 0 and epoch%1 ==0:
        print ('Test Epoch: {}'.format(epoch))
        print ('Test Epoch: {}'.format(epoch),file=test_log_file)
        # testing
        cmc, mAP = test(epoch)

        print('FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[9], cmc[19], mAP))
        print('FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[9], cmc[19], mAP), file = test_log_file)
        test_log_file.flush()
        
        # save model
        if cmc[0] > best_acc: # not the real best for sysu-mm01 
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
        
        # save model every 20 epochs    
        if epoch > 1 and epoch%args.save_epoch ==0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
            
            
            