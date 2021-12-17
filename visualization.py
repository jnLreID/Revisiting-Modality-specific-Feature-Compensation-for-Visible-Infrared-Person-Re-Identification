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
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData1, RegDBData1, TestData1
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
# from model.fcmodeln85 import embed_net
from model.fcmodeln85reg import embed_net
from utils import *
import time 
import scipy.io as scio
import numpy as np
from PIL import Image
import cv2
from os.path import dirname as ospdn
from os.path import join as ospj
import os 
################################################
def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not os.path.exists(path):
    os.makedirs(path)

####################################################
def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist
#####################################################
def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
#############################################################
def get_rank_list(dist_vec, q_id, q_cam, g_ids, g_cams, rank_list_size):
  """Get the ranking list of a query image
  Args:
    dist_vec: a numpy array with shape [num_gallery_images], the distance
      between the query image and all gallery images
    q_id: a scalar, query id
    q_cam: a scalar, query camera
    g_ids: a numpy array with shape [num_gallery_images], gallery ids
    g_cams: a numpy array with shape [num_gallery_images], gallery cameras
    rank_list_size: a scalar, the number of images to show in a rank list
  Returns:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
  """
  sort_inds = np.argsort(dist_vec)
  rank_list = []
  same_id = []
  i = 0
  for ind, g_id, g_cam in zip(sort_inds, g_ids[sort_inds], g_cams[sort_inds]):
    # Skip gallery images with same id and same camera as query
    if (q_id == g_id) and (q_cam == g_cam):
      continue
    same_id.append(q_id == g_id)
    rank_list.append(ind)
    i += 1
    if i >= rank_list_size:
      break
  return rank_list, same_id

def get_rank_list_regdb(dist_vec, q_id, q_cam, g_ids, g_cams, rank_list_size):
  """Get the ranking list of a query image
  Args:
    dist_vec: a numpy array with shape [num_gallery_images], the distance
      between the query image and all gallery images
    q_id: a scalar, query id
    q_cam: a scalar, query camera
    g_ids: a numpy array with shape [num_gallery_images], gallery ids
    g_cams: a numpy array with shape [num_gallery_images], gallery cameras
    rank_list_size: a scalar, the number of images to show in a rank list
  Returns:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
  """
  sort_inds = np.argsort(dist_vec)
  rank_list = []
  same_id = []
  i = 0
  for ind, g_id, g_cam in zip(sort_inds, g_ids[sort_inds], g_cams[sort_inds]):
    # Skip gallery images with same id and same camera as query
#    if q_id == g_id:
#      continue
    same_id.append(q_id == g_id)
    rank_list.append(ind)
    i += 1
    if i >= rank_list_size:
      break
  return rank_list, same_id
#############################################################
def add_border(im, border_width, value):
  """Add color border around an image. The resulting image size is not changed.
  Args:
    im: numpy array with shape [3, im_h, im_w]
    border_width: scalar, measured in pixel
    value: scalar, or numpy array with shape [3]; the color of the border
  Returns:
    im: numpy array with shape [3, im_h, im_w]
  """
  assert (im.ndim == 3) and (im.shape[0] == 3)
  im = np.copy(im)

  if isinstance(value, np.ndarray):
    # reshape to [3, 1, 1]
    value = value.flatten()[:, np.newaxis, np.newaxis]
  im[:, :border_width, :] = value
  im[:, -border_width:, :] = value
  im[:, :, :border_width] = value
  im[:, :, -border_width:] = value

  return im
def make_im_grid(ims, n_rows, n_cols, space, pad_val):
  """Make a grid of images with space in between.
  Args:
    ims: a list of [3, im_h, im_w] images
    n_rows: num of rows
    n_cols: num of columns
    space: the num of pixels between two images
    pad_val: scalar, or numpy array with shape [3]; the color of the space
  Returns:
    ret_im: a numpy array with shape [3, H, W]
  """
  assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
  assert len(ims) <= n_rows * n_cols
  h, w = ims[0].shape[1:]
  H = h * n_rows + space * (n_rows - 1)
  W = w * n_cols + space * (n_cols - 1)
  if isinstance(pad_val, np.ndarray):
    # reshape to [3, 1, 1]
    pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
  ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
  for n, im in enumerate(ims):
    r = n // n_cols
    c = n % n_cols
    h1 = r * (h + space)
    h2 = r * (h + space) + h
    w1 = c * (w + space)
    w2 = c * (w + space) + w
    ret_im[:, h1:h2, w1:w2] = im
  return ret_im

def read_im(im_path):
  # shape [H, W, 3]
  im = np.asarray(Image.open(im_path))
  # Resize to (im_h, im_w) = (128, 64)
  resize_h_w = (128, 64)
  if (im.shape[0], im.shape[1]) != resize_h_w:
    im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
  # shape [3, H, W]
  im = im.transpose(2, 0, 1)
  return im

def save_im(im, save_path):
  """im: shape [3, H, W]"""
  may_make_dir(ospdn(save_path))
  im = im.transpose(1, 2, 0)
  Image.fromarray(im).save(save_path)
  
def save_rank_list_to_im(rank_list, same_id, q_im_path, g_im_paths, save_path):
  """Save a query and its rank list as an image.
  Args:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
    q_im_path: query image path
    g_im_paths: ALL gallery image paths
    save_path: path to save the query and its rank list as an image
  """
  ims = [read_im(q_im_path)]
  for ind, sid in zip(rank_list, same_id):
    im = read_im(g_im_paths[ind])
    # Add green boundary to true positive, red to false positive
    color = np.array([0, 255, 0]) if sid else np.array([255, 0, 0])
    im = add_border(im, 3, color)
    ims.append(im)
  im = make_im_grid(ims, 1, len(rank_list) + 1, 8, 255)
  save_im(im, save_path)
#############################################################


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model10.51/', type=str, help='model save path')
# parser.add_argument('--model_path', default='reg_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='Method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(1)
dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data/RGBT-reid/SYSU_MM011/'
    
    n_class = 395
    test_mode = [2, 1] 
elif dataset =='regdb':
    data_path = '/data/RGBT-reid/RegDB/'
    n_class = 206
    test_mode = [1, 2]
 
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
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))



print('==> Loading data..')
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
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)

      
elif dataset =='regdb':
    # training set
    trainset = RegDBData1(data_path, args.trial, transform=None)
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

queryset = TestData1(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))   
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = 3072

if args.arch =='resnet50':
    pool_dim = 3072
elif args.arch =='resnet18':
    pool_dim = 512 *12

def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feature_dim))
    gall_feat_pool = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            feat_pool, feat = net(input,input,input2,input2,test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_pool
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feature_dim))
    query_feat_pool = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input,input2,label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input2 = Variable(input2.cuda())
            feat_pool, feat = net(input,input,input2,input2,test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat, query_feat_pool 
    
query_feat, query_feat_pool = extract_query_feat(query_loader)    

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_gall_feat(gall_loader)
    # fc feature 
#    distmat = np.matmul(query_feat, np.transpose(gall_feat))
#    cmc, mAP  = eval_regdb(-distmat, query_label, gall_label)
#    print(cmc[0], cmc[4], cmc[9], mAP)
    save_paths = [ospj('rank_lists', n[-7:]) for n in query_img]
    # pool5 feature
    distmat = compute_dist(query_feat, gall_feat, type='euclidean')
    for dist_vec, q_id, q_cam, q_im_path, save_path in zip(
          distmat, query_label, query_label, query_img, save_paths):

        rank_list, same_id = get_rank_list_regdb(
          dist_vec, q_id, q_cam, gall_label, gall_label, 10)
        
        save_rank_list_to_im(rank_list, same_id, q_im_path, gall_img, save_path)
    
elif dataset =='sysu':
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)
    
    trial_gallset = TestData1(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
    trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    
    gall_feat, gall_feat_pool = extract_gall_feat(trial_gall_loader)
    
    # fc feature 
    distmat = compute_dist(query_feat, gall_feat, type='euclidean')
    save_paths = [ospj('rank_lists', str(n)+'.jpg') for n in range(len(query_img))]
    # pool5 feature
    distmat_pool = compute_dist(query_feat_pool, gall_feat_pool, type='euclidean')
    ii = 0
    for dist_vec, q_id, q_cam, q_im_path, save_path in zip(
          distmat, query_label, query_cam, query_img, save_paths):
        rank_list, same_id = get_rank_list(
          dist_vec, q_id, q_cam, gall_label, gall_cam, 10)
        
        save_rank_list_to_im(rank_list, same_id, q_im_path, gall_img, save_path)
        ii = ii+1
        if ii == 5000:
            break