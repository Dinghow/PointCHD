#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM

Modified by 
@Author: Dinghao Yang
@Contact: dinghowyang@gmail.com
@Time: 2023/03/30 16:30 PM
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, args=None, idx=None, dim9=False, concat=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    if args is None:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    if concat:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (batch_size, 2*num_dims, num_points, k)
    else:
        feature = (feature-x).permute(0, 3, 1, 2) # (batch_size, num_dims, num_points, k)
  
    return feature      


def manifold_projection(x, nor, k=20, args=None):
    # x (batch_size, 3, num_points), nor (batch_size, 3, num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    nor = nor.view(batch_size, -1, num_points)
    
    idx = knn(x, k=k)   # (batch_size, num_points, k)

    if args is None:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() 
    nor = nor.transpose(2, 1).contiguous()
    nor_groups = nor.view(batch_size*num_points, -1)[idx, :]
    nor_groups = nor_groups.view(batch_size, num_points, k, num_dims)

    batch_similarity = []
    for i in range(0, batch_size):
        sample_similarity = []
        for j in range(0, num_points):
            centroid_normal = nor_groups[i][j][:1]
            neighbor_normals = nor_groups[i][j][:]
            similarity = torch.cosine_similarity(centroid_normal.unsqueeze(1), neighbor_normals.unsqueeze(0), dim=2)
            sample_similarity.append(similarity)
        sample_similarity = torch.cat(sample_similarity, dim=0)
        batch_similarity.append(sample_similarity.unsqueeze(0))
    batch_similarity = torch.cat(batch_similarity, dim=0)

    #  (b, num_points, k) 
    return batch_similarity


class PointManifold_NNML(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointManifold_NNML, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn0_0 = nn.BatchNorm1d(2)
        self.bn0_1 = nn.BatchNorm1d(2)
        self.bn0_2 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        
        self.conv0_0 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0_1 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0_2 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_2,
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0) # x (b, 3, n)
        # x = LLE(x, 12, 2)
        x_2d_z = self.conv0_0(x[:,:2,:]) # (b, 2, n)
        x_2d_z = torch.mul(x_2d_z, x[:, 2, :].reshape(batch_size, 1, -1)) # (b, 2, n) x (b, 1, n)
        x_2d_y = self.conv0_1(x[:,[0,2],:])
        x_2d_y = torch.mul(x_2d_y, x[:, 1, :].reshape(batch_size, 1, -1))
        x_2d_x = self.conv0_1(x[:,1:3,:])
        x_2d_x = torch.mul(x_2d_x, x[:, 0, :].reshape(batch_size, 1, -1))
        x = torch.cat((x, x_2d_x, x_2d_y, x_2d_z), 1) # (b, 9, n)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 7, num_points) -> (batch_size, 7x2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 7*2, num_points, k) -> (batch_size, 128, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 128, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 128+128+256+512, num_points)

        x = self.conv5(x)                       # (batch_size, 128+128+256+512, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


class PointManifold_NNML_partseg_pointchd(nn.Module):
    def __init__(self, args, seg_num_all):
        super(PointManifold_NNML_partseg_pointchd, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        
        self.bn0_0 = nn.BatchNorm1d(2)
        self.bn0_1 = nn.BatchNorm1d(2)
        self.bn0_2 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv0_0 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0_1 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0_2 = nn.Sequential(nn.Conv1d(2, 2, kernel_size = 1, bias=False),
                                   self.bn0_2,
                                   nn.LeakyReLU(negative_slope=0.2)) 
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64*3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+64*3, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x_2d_z = self.conv0_0(x[:,:2,:]) # (b, 2, n)
        x_2d_z = torch.mul(x_2d_z, x[:, 2, :].reshape(batch_size, 1, -1)) # (b, 2, n) x (b, 1, n)
        x_2d_y = self.conv0_1(x[:,[0,2],:])
        x_2d_y = torch.mul(x_2d_y, x[:, 1, :].reshape(batch_size, 1, -1))
        x_2d_x = self.conv0_1(x[:,1:3,:])
        x_2d_x = torch.mul(x_2d_x, x[:, 0, :].reshape(batch_size, 1, -1))
        x = torch.cat((x, x_2d_x, x_2d_y, x_2d_z), 1) # (b, 9, n)
        x = get_graph_feature(x, k=self.k, args=self.args)      # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, args=self.args)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, args=self.args)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv8(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv9(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class PointManifold2(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointManifold2, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv_mp1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(3),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_mp2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_mp3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_mp4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, xyznxnynz):
        batch_size = xyznxnynz.size(0)
        num_points = xyznxnynz.size(2)
        x = xyznxnynz[:, :3, :]
        nor = xyznxnynz[:, 3:, :]

        weight = manifold_projection(x, nor, k=self.k, args=self.args)  #(batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        n_f1 = get_graph_feature(x, k=self.k, args=self.args, concat=False)      # (batch_size, 3, num_points) -> (batch_size, 3, num_points, k)
        w1 =  weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f1.size(1)).permute(0, 3, 1, 2) # (batch_size, num_points, k) -> (batch_size, 3, num_points, k) 
        n_f1 = torch.mul(n_f1, self.conv_mp1(w1))
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f1.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f1, x), dim=1).contiguous() 
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        n_f2 = get_graph_feature(x1, k=self.k, args=self.args, concat=False)     # (batch_size, 64, num_points) -> (batch_size, 64, num_points, k)
        w2 = weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f2.size(1)).permute(0, 3, 1, 2)
        n_f2 = torch.mul(n_f2, self.conv_mp2(w2))
        x = x1.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f2.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f2, x), dim=1).contiguous()   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        n_f3 = get_graph_feature(x2, k=self.k, args=self.args, concat=False)     # (batch_size, 64, num_points) -> (batch_size, 64, num_points, k)
        w3 = weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f3.size(1)).permute(0, 3, 1, 2)
        n_f3 = torch.mul(n_f3, self.conv_mp3(w3))
        x = x2.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f3.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f3, x), dim=1).contiguous()    # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        n_f4 = get_graph_feature(x3, k=self.k, args=self.args, concat=False)     # (batch_size, 64, num_points) -> (batch_size, 64, num_points, k)
        w4 = weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f4.size(1)).permute(0, 3, 1, 2)
        n_f4 = torch.mul(n_f4, self.conv_mp4(w4))
        x = x3.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f4.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f4, x), dim=1).contiguous()    # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


class PointManifold2_partseg_pointchd(nn.Module):
    def __init__(self, args, seg_num_all):
        super(PointManifold2_partseg_pointchd, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv_mp1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(3),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_mp2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_mp3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64*3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+64*3, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, xyznxnynz):
        batch_size = xyznxnynz.size(0)
        num_points = xyznxnynz.size(2)
        x = xyznxnynz[:, :3, :]
        nor = xyznxnynz[:, 3:, :]

        weight = manifold_projection(x, nor, k=self.k, args=self.args)      # (batch_size, 3, num_points) -> (batch_size, num_points, k)
        n_f1 = get_graph_feature(x, k=self.k, args=self.args, concat=False)      # (batch_size, 3, num_points) -> (batch_size, 3, num_points, k)
        w1 =  weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f1.size(1)).permute(0, 3, 1, 2) # (batch_size, num_points, k) -> (batch_size, 3, num_points, k) 
        n_f1 = torch.mul(n_f1, self.conv_mp1(w1))
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f1.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f1, x), dim=1).contiguous()         # (batch_size, 3, num_points, k) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        n_f2 = get_graph_feature(x1, k=self.k, args=self.args, concat=False)     # (batch_size, 64, num_points) -> (batch_size, 64, num_points, k)
        w2 = weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f2.size(1)).permute(0, 3, 1, 2)
        n_f2 = torch.mul(n_f2, self.conv_mp2(w2))
        x = x1.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f2.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f2, x), dim=1).contiguous()
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        n_f3 = get_graph_feature(x2, k=self.k, args=self.args, concat=False)     # (batch_size, 64, num_points) -> (batch_size, 64, num_points, k)
        w3 = weight.view(batch_size, num_points, self.k, 1).repeat(1, 1, 1, n_f3.size(1)).permute(0, 3, 1, 2)
        n_f3 = torch.mul(n_f3, self.conv_mp3(w3))
        x = x2.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, n_f3.size(1)).repeat(1, 1, self.k, 1).permute(0, 3, 1, 2)
        x = torch.cat((n_f3, x), dim=1).contiguous()
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv8(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv9(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x