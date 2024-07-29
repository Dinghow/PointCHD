#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM
"""


import os
import h5py
import numpy as np
import json
from torch.utils.data import Dataset
import open3d as o3d
import math


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def sampling_points(points, target_num_points):
    num_points = points.shape[0]
    if num_points/target_num_points >= 2:
        pcd = o3d.geometry.PointCloud()
        if points.shape[1] == 3:
            pcd.points = o3d.utility.Vector3dVector(points)
        elif points.shape[1] == 6:
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.normals = o3d.utility.Vector3dVector(points[:, 3:])
        else:
            raise("Invalid point cloud dimension!")
        pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, math.floor(num_points/target_num_points))
        pcd_new_np = np.concatenate((np.asarray(pcd_new.points), np.asarray(pcd_new.normals)), axis=1)
    else:
        sampling_idx = np.random.choice(num_points, target_num_points, replace=False)
        pcd_new_np = points[sampling_idx]
    return pcd_new_np
    

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class PointCHD(Dataset):
    def __init__(self, num_points, partition='train', task='part_seg', data_type='xyz', norm=False, seg_result=''):
        if 'train' not in partition and 'val' not in partition:
            raise('Wrong dataset split')
        assert task in ['part_seg', 'cls']
        assert data_type in ['xyz', 'xyznxnynz']
        assert num_points in [1024, 2048, 4096]
        self.num_points = num_points
        self.partition = partition        
        self.task = task
        self.seg_num_all = 7
        self.norm = norm
        self.data_type = data_type
        self.seg_result = seg_result

        if self.seg_result == '': # only support validation set currently
            self.data, self.label, self.ids = self._load_data_partseg(partition)
        else:
            ori_data, ori_label, ori_ids = self._load_data_partseg(partition)
            self.data, self.label, self.ids = self._load_part_seg_result(ori_data, ori_label, ori_ids)

        print('Number of samples: {:d}\nNumber of points {:d}'.format(len(self.data), self.num_points))

    def _load_data_partseg(self, partition):
        BASE_DIR = "/home/dinghow/Documents/PointCHD/data_"+self.data_type+"/new_split"
        # filepath = os.path.join(BASE_DIR, partition+'_wo_Myo_'+str(self.num_points)+'.h5')
        filepath = os.path.join(BASE_DIR, partition+'_'+str(self.num_points)+'.h5')
        f = h5py.File(filepath, 'r+')
        data = f['data'][:].astype('float32')
        ids = f['ids'][:].astype('int64')
        cls_label = f['class'][:].astype('float32') 
        seg_label = (f['seg'][:]-1).astype('int64') # part seg label:1-7 -> 0-6
        if self.data_type == 'xyznxnynz':
            normal = f['normal'][:].astype('float32')
            data = np.concatenate((data, normal), axis=2)
        f.close()

        if self.task == 'part_seg':
            return data, seg_label, ids
        else: 
            return data, cls_label, ids

    def _construct_cls_id_dict(self, ids, cls_label):
        id2cls = {}
        assert ids.shape[0] == cls_label.shape[0]
        for i in range(0, ids.shape[0]):
            id2cls[ids[i]] = cls_label[i]
        return id2cls

    def _load_part_seg_result(self, ori_data, ori_label, ori_ids): # load data without Myo part from part segmentation results
        seg_result_dict = json.load(open(self.seg_result, 'r'))
        id2cls = self._construct_cls_id_dict(ori_ids, ori_label)
        new_data = []
        new_ids = []
        new_cls_label = []

        for sample in seg_result_dict:
            data = np.array(sample['data'])
            seg_label = np.array(sample['seg'])
            gt_label = np.array(sample['gt'])
            part_idx = ~np.array(seg_label == 4)
            part_data = data[part_idx, :]
            num_part_points = part_idx.sum()

            # sampling the points from seg results, and combined with gt part points if not satifify the target num points
            if num_part_points > self.num_points:
                part_data = sampling_points(part_data, self.num_points)
            elif num_part_points < self.num_points:
                part_idx_gt = ~np.array(gt_label == 4)
                part_data_gt = data[part_idx_gt, :]
                num_part_points_gt = part_idx_gt.sum()
                sampled_gt_points = sampling_points(part_data_gt, self.num_points-num_part_points)
                part_data = np.concatenate((part_data, sampled_gt_points), axis=0)
            part_data = part_data[:self.num_points,]

            new_data.append(part_data)
            new_ids.append(sample['id'])
            new_cls_label.append(id2cls[sample['id']])

        new_data = np.array(new_data).astype('float32')
        new_ids = np.array(new_ids).astype('int64')
        new_cls_label = np.array(new_cls_label).astype('float32')
        
        return new_data, new_cls_label, new_ids
            
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.norm:
            pointcloud_norm = pc_normalize(pointcloud)
            pointcloud = np.concatenate((pointcloud, pointcloud_norm), axis=1)
        ids = self.ids[item]
        if self.task == 'part_seg':
            label = self.label[item][:self.num_points]
            if self.partition == 'train':
                # pointcloud = translate_pointcloud(pointcloud)
                indices = list(range(pointcloud.shape[0]))
                np.random.shuffle(indices)
                pointcloud = pointcloud[indices]
                label = label[indices]
        else:
            label = self.label[item]
            if self.partition == 'train':
                # pointcloud = translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)
        return pointcloud, label, ids

    def __len__(self):
        return self.data.shape[0]
