from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from math import sin, cos
from einops import rearrange, repeat

def init_weights(m):
    class_name=m.__class__.__name__

    if "Conv2d" in class_name or "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/smeetrs/deep_avsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Gumbel


class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch

#
# class VideoNet(nn.Module):
#
#     """
#     A video-only speech transcription model based on the Transformer architecture.
#     Architecture: A stack of 12 Transformer encoder layers,
#                   first 6 form the Encoder and the last 6 form the Decoder.
#     Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
#     Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
#     Output: Log probabilities over the character set at each time step.
#     """
#
#     def __init__(self, dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout):
#         super(VideoNet, self).__init__()
#         self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
#         encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
#         self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
#         self.videoDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
#         return
#
#
#     def forward(self, inputBatch):
#         #batch = self.positionalEncoding(inputBatch)
#         batch = inputBatch
#         batch1 = self.videoEncoder(batch)
#         # batch1 = self.videoDecoder(batch)
#         batch = batch1.mean(dim=-1)
#         _,index = torch.topk(batch,2,dim=1)
#         # print(index.shape)
#         # out = index.detach().cpu().tolist()
#
#         return index,batch1

class VideoNet(nn.Module):

    """
    A video-only speech transcription model based on the Transformer architecture.
    Architecture: A stack of 12 Transformer encoder layers,
                  first 6 form the Encoder and the last 6 form the Decoder.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout):
        super(VideoNet, self).__init__()
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        return

    def gumbel_top_k(self, x, k, tau=0.5 ,eps=1e-10):

        x = F.relu(x)

        gumbels = Gumbel(0, 1).sample(x.shape).cuda()
        y = (gumbels + torch.log(x + eps)) / tau
        # 这里需要进行两次softmax吗？因为之前已经进行过一次softmax操作了
        y = y.mean(dim=2)
        # 修改这里,不直接使用topk
        # print(y)
        y = F.softmax(y, dim=1)

        indices = torch.multinomial(y, k, replacement=False)
        return indices


    def forward(self, inputBatch):
        #batch = self.positionalEncoding(inputBatch)
        batch = inputBatch
        batch1 = self.videoEncoder(batch)
        index = self.gumbel_top_k(batch1, 3)
        # batch1 = self.videoDecoder(batch)
        # batch = batch1.mean(dim=-1)
        # _,index = torch.topk(batch,2,dim=1)
        # print(index.shape)
        # out = index.detach().cpu().tolist()

        return index,batch1


class BTwins(nn.Module):

    @ex.capture
    def __init__(self, hidden_size, lambd, pj_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )
        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.lambd = lambd
        self.projector1 = nn.Linear(hidden_size,8)


    def forward(self, feat1, feat2):

        feat1 = self.projector(feat1)
        feat2 = self.projector(feat2)
        feat1_norm = self.bn(feat1)
        feat2_norm = self.bn(feat2)

        N, D = feat1_norm.shape
        c = (feat1_norm.T @ feat2_norm).div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        BTloss = on_diag + self.lambd * off_diag

        return BTloss 

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTwins_Linear(nn.Module):

    @ex.capture
    def __init__(self, hidden_size, lambd, pj_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.projector1 = nn.Linear(hidden_size,8)


    def forward(self, feat1, feat2):

        feat1 = self.projector1(feat1)
        return feat1


    
@ex.capture 
def get_stream(data, view):
    N, C, T, V, M = data.shape

    if view == 'joint':
        pass

    elif view == 'motion':
        motion = torch.zeros_like(data)
        motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

        data = motion

    elif view == 'bone':
        Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        bone = torch.zeros_like(data)

        for v1, v2 in Bone:
            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

        data = bone
    
    else:

        return None

    return data

@ex.capture
def shear(input_data, shear_amp):
    # n c t v m
    temp = input_data.clone()
    amp = shear_amp
    Shear       = np.array([
                    [1, random.uniform(-amp, amp), 	0],
                    [random.uniform(-amp, amp), 1, 	0],
                    [random.uniform(-amp, amp), 	random.uniform(-amp, amp),1]
                    ])
    Shear = torch.Tensor(Shear).cuda()
    output =  torch.einsum('n c t v m, c d -> n d t v m',[temp,Shear])

    return output

def reverse(data,p=0.5):

    N,C,T,V,M = data.shape
    temp = data.clone()

    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp
        
@ex.capture
def crop(data, temperal_padding_ratio=6):
    input_data = data.clone()
    N, C, T, V, M = input_data.shape
    #padding
    padding_len = T // temperal_padding_ratio
    frame_start = torch.randint(0, padding_len * 2 + 1,(1,))
    first_clip = torch.flip(input_data[:,:,:padding_len],dims=[2])
    second_clip = input_data
    thrid_clip = torch.flip(input_data[:,:,-padding_len:],dims=[2])
    out = torch.cat([first_clip,second_clip,thrid_clip],dim=2)
    out = out[:, :, frame_start:frame_start + T]

    return out

def random_rotate(data):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        R = torch.Tensor(R).cuda()
        output =  torch.einsum('n c t v m, c d -> n d t v m',[seq,R])
        return output

    # n c t v m
    new_seq = data.clone()
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    return new_seq

@ex.capture
def get_ignore_joint(mask_joint):

    ignore_joint = random.sample(range(25), mask_joint)
    print('get_ignore_joint',ignore_joint)
    return ignore_joint

@ex.capture
def get_ignore_part(mask_part):

    # left_hand = [8,9,10,11,23,24]
    # right_hand = [4,5,6,7,21,22]
    # left_leg = [16,17,18,19]
    # right_leg = [12,13,14,15]
    # body = [0,1,2,3,20]
    # all_joint = [left_hand, right_hand, left_leg, right_leg, body]
    # part = random.sample(range(5), mask_part)
    # ignore_joint = []
    # for i in part:
    #     ignore_joint += all_joint[i]
    # print('get_ignore_part',ignore_joint)

    left_hand = [6,8,10]
    right_hand = [5,7,9]
    left_leg = [12,14,16]
    right_leg = [11,13,15]
    body = [0,1,2,3,4]
    all_joint = [left_hand, right_hand, left_leg, right_leg, body]
    part = random.sample(range(5), mask_part)
    ignore_joint = []
    for i in part:
        ignore_joint += all_joint[i]
    print('get_ignore_part',ignore_joint)

    return ignore_joint

def gaus_noise(data, mean= 0, std = 0.01):
    temp = data.clone()
    n, c, t, v, m = temp.shape
    noise = np.random.normal(mean, std, size=(n, c, t, v, m))
    noise = torch.Tensor(noise).cuda()

    return temp + noise

def gaus_filter(data):
    temp = data.clone()
    g = GaussianBlurConv(3).cuda()
    return g(temp)

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        kernel =  kernel.float()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,5)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.weight = self.weight.cuda()

        prob = np.random.random_sample()
        if prob < 0.5:
            #x = x.permute(3,0,2,1) # M,C,V,T
            x = rearrange(x, 'n c t v m -> (n m) c v t')
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            #x = x.permute(1,-1,-2, 0) #C,T,V,M
            x = rearrange(x, '(n m) c v t -> n c t v m', m = 2)

        return x

@ex.capture
def temporal_cropresize(input_data,max_frame,output_size,l_ratio=[0.1,1]):

    num_of_frames = max_frame

    n, c, t, v, m = input_data.shape
    min_crop_length = 64
    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)
    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:, :,start:start+temporal_crop_length, :, :]
    temporal_context = rearrange(temporal_context,'n c t v m -> n (c v m) t')
    temporal_context=temporal_context[: , :, :,None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=-1)
    temporal_context = rearrange(temporal_context,'n (c v m) t -> n c t v m',c=c,v=v,m=m)
    return temporal_context

def random_spatial_flip(data, p=0.5):
    temp = data.clone()
    # order = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16,
    # 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
    order = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
    if random.random() < p:
        temp = temp[:, :, :, order, :]

    return temp

def random_time_flip(temp, p=0.5):
    # temp = data.clone()
    T = temp.shape[2]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp

@ex.capture
def motion_att_temp_big_mask(data, mask_frame):
    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=False)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    return output

@ex.capture
def motion_att_temp_local_big_mask_score(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # joint_mask
    joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    temp_joint = temp.gather(3,joint_list)

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    temp_joint_score = temp_joint[:,2,:,:,:]
    # print(temp_joint_score,temp_joint_score.shape)
    temp_joint_score = repeat(temp_joint_score,'n t v m -> n c t v m', c=c)
    # print('a',temp_joint_score.shape)
    temp_joint = temp_joint.mul(temp_joint_score)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=False)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    return output


@ex.capture
def motion_att_temp_local_small_mask_score(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # joint_mask
    joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    temp_joint = temp.gather(3,joint_list)

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    temp_joint_score = temp_joint[:,2,:,:,:]
    # print(temp_joint_score,temp_joint_score.shape)
    temp_joint_score = repeat(temp_joint_score,'n t v m -> n c t v m', c=c)
    # print('a',temp_joint_score.shape)
    temp_joint = temp_joint.mul(temp_joint_score)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=True)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    return output


@ex.capture
def motion_att_temp_global_small_mask_score(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # # joint_mask
    # joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    # joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    # temp_joint = temp.gather(3,joint_list)
    temp_joint = temp

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    temp_joint_score = temp_joint[:,2,:,:,:]
    # print(temp_joint_score,temp_joint_score.shape)
    temp_joint_score = repeat(temp_joint_score,'n t v m -> n c t v m', c=c)
    # print('a',temp_joint_score.shape)
    temp_joint = temp_joint.mul(temp_joint_score)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=True)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    return output


@ex.capture
def motion_att_temp_global_big_mask_score(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # # joint_mask
    # joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    # joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    # temp_joint = temp.gather(3,joint_list)
    temp_joint = temp

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    temp_joint_score = temp_joint[:,2,:,:,:]
    # print(temp_joint_score,temp_joint_score.shape)
    temp_joint_score = repeat(temp_joint_score,'n t v m -> n c t v m', c=c)
    # print('a',temp_joint_score.shape)
    temp_joint = temp_joint.mul(temp_joint_score)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=False)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    return output

@ex.capture
def motion_att_temp_small_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    # motion = torch.zeros_like(temp)
    # motion[:, :-1, :-1, :, :] = temp[:, :-1, 1:, :, :] - temp[:, :-1, :-1, :, :]
    # motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample

    # ## random temp mask
    # random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    # random_frame.sort()
    # output = temp_resample[:, :, random_frame, :, :]

    return output

@ex.capture
def people_att_temp_small_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = m - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :, :, :-1] = temp[:, :, :, :, 1:] - temp[:, :, :, :, :-1]
    motion = -(motion)**2
    temporal_att = motion.mean((1,2,3))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list1 = repeat(temp_list,'n m -> n c t v m',c=c,t=t,v=v)


    temp_resample = temp.gather(4,temp_list1)
    # ## random temp mask
    # random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    # random_frame.sort()
    # output = temp_resample[:, :, random_frame, :, :]

    return temp_resample,temp_list

@ex.capture
def motion_att_temp_local_small_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # joint_mask
    joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    temp_joint = temp.gather(3,joint_list)

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample


    return output

@ex.capture
def motion_att_temp_local_big_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    # joint_mask
    joint_save = torch.tensor([2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    joint_list = repeat(joint_save, 'v -> n c t v m', n=n, c=c, t=t, m=m)
    temp_joint = temp.gather(3,joint_list)

    ## get the motion_attention value
    # motion = torch.zeros_like(temp)
    # motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]

    motion = torch.zeros_like(temp_joint)
    motion[:, :-1, :-1, :, :] = temp_joint[:, :-1, 1:, :, :] - temp_joint[:, :-1, :-1, :, :]
    motion = motion[:,:-1,:,:,:]

    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num, largest=False)

    temp_list,_ = torch.sort(temp_list.squeeze())

    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)

    temp_resample = temp.gather(2,temp_list)
    output = temp_resample


    return output


@ex.capture
def motion_att_temp_mask2(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = m - 6

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :, :, :-1] = temp[:, :, :, :, 1:] - temp[:, :, :, :, :-1]
    motion = -(motion)**2
    temporal_att = motion.mean((1,2,3))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n m -> n c t v m',c=c,t=t,v=v)


    temp_resample = temp.gather(4,temp_list)

    return temp_resample

@ex.capture
def motion_att_temp_mask3(data, mask_frame):
    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = m - 6

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :, :, :-1] = temp[:, :, :, :, 1:] - temp[:, :, :, :, :-1]
    motion = (motion)**2
    temporal_att = motion.mean((1,2,3))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n m -> n c t v m',c=c,t=t,v=v)
    temp_resample = temp.gather(4,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-3)
    random_frame.sort()
    output = temp_resample[:, :, :, :, random_frame]

    return output

@ex.capture
def motion_att_temp_mask4(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = m - 6

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :, :, :-1] = temp[:, :, :, :, 1:] - temp[:, :, :, :, :-1]
    motion = -(motion)**2
    temporal_att = motion.mean((1,2,3))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list2 = temp_list
    temp_list = repeat(temp_list,'n m -> n c t v m',c=c,t=t,v=v)


    temp_resample = temp.gather(4,temp_list)


    return temp_resample,temp_list2


@ex.capture
def central_spacial_mask(mask_joint):

    # Degree Centrality
    # degree_centrality = [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    #                     2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 1, 2]
    # all_joint = []
    # for i in range(25):
    #     all_joint += [i]*degree_centrality[i]
    # degree_centrality = [4,2,2,1,1,4,4,2,2,1,1,3,3,2,2,1,1]
    degree_centrality = [1,1,3,5,5,3,5,5,3,4,4,3,4,4,1,1,1,1] # 对手的mask更高，对脚也要有比较高的mask概率
    all_joint = []
    for i in range(18):
        all_joint += [i]*degree_centrality[i]


    ignore_joint = random.sample(all_joint, mask_joint)

    return ignore_joint

#
# def semi_mask(mask_num):
#
#     p = random.random()
#     if p<0.5:
#         ignore_joint = central_spacial_mask(mask_num)
#     else:
#         ignore_joint = []
#
#     return ignore_joint