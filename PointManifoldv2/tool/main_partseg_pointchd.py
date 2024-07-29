#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM

Modified by 
@Author: Dinghao Yang
@Contact: dinghowyang@gmail.com
@Time: 2023/03/30 16:30 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data import PointCHD
from model.main_models import PointManifold_NNML_partseg_pointchd, PointManifold2_partseg_pointchd
import numpy as np
from torch.utils.data import DataLoader
from util.util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
import json

part_colormap = [
    {"id": 1, "label": "LV", "color": [152,223,138]},
    {"id": 2, "label": "RV", "color": [174,199,232]},
    {"id": 3, "label": "LA", "color": [255,105,180]},
    {"id": 4, "label": "RA", "color": [31,119,180]},
    {"id": 5, "label": "Myo", "color": [112,128,144]},
    {"id": 6, "label": "AO", "color": [96,207,209]},
    {"id": 7, "label": "PA", "color": [227,119,194]}
]

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(7)
    U_all = np.zeros(7)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(7):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all 


def calculate_part_IoU(pred_np, seg_np):
    parts = range(7)
    part_ious = []
    for shape_idx in range(seg_np.shape[0]):
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
    return part_ious


def visualization(visu_format, data, pred, seg, ids):
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'):
            os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/')
        for j in range(0, data.shape[2]):
            RGB.append(part_colormap[int(pred[i][j])]['color'])
            RGB_gt.append(part_colormap[int(seg[i][j])]['color'])
        pred_np = []
        seg_np = []
        pred_np.append(pred[i].cpu().numpy())
        seg_np.append(seg[i].cpu().numpy())
        sample_id = int(ids[i])
        xyz_np = data[i].cpu().numpy()
        xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
        xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
        filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+str(sample_id)+'_pred.'+visu_format
        filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+str(sample_id)+'_gt.'+visu_format
        if visu_format=='txt':
            np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
            np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
            print('TXT visualization file saved in', filepath)
            print('TXT visualization file saved in', filepath_gt)
        elif visu_format=='ply':
            xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
            xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(filepath)
            vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(filepath_gt)
            print('PLY visualization file saved in', filepath)
            print('PLY visualization file saved in', filepath_gt)
        else:
            print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
            (visu_format))
            exit()


def train(args, io):
    train_dataset = PointCHD(partition='train', num_points=args.num_points, task='part_seg', data_type=args.data_type, norm=args.data_norm)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(PointCHD(partition='val', num_points=args.num_points, task='part_seg', data_type=args.data_type, norm=args.data_norm), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    if args.model == 'dgcnn':
        model = DGCNN_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointnet':
        model = PointNet_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointmanifold':
        model = PointManifold_NNML_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointmanifold2':
        model = PointManifold2_partseg_pointchd(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    if args.cuda:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        for data, seg, ids in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg, ids in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(PointCHD(partition='val', num_points=args.num_points, task='part_seg', data_type=args.data_type, norm=args.data_norm),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    # partseg_colors = test_loader.dataset.partseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointnet':
        model = PointNet_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointmanifold':
        model = PointManifold_NNML_partseg_pointchd(args, seg_num_all).to(device)
    elif args.model == 'pointmanifold2':
        model = PointManifold2_partseg_pointchd(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_ids = []
    test_data = []
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    for data, seg, ids in test_loader:
        data, seg = data.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        data_np = data.permute(0, 2, 1).cpu().numpy()
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_data.append(data_np)
        test_ids.append(ids.cpu().numpy())
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        # visiualization
        if args.visu:
            visualization(args.visu_format, data, pred, seg, ids) 
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_data = np.concatenate(test_data, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
    test_parts_ious = calculate_part_IoU(test_pred_seg, test_true_seg)
    if args.save_res:
        pred_data = test_data
        pred_seg = test_pred_seg
        gt_seg = test_true_seg
        pred_id = test_ids
        result_dir = os.path.join('outputs', args.exp_name, 'result')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_filepath = os.path.join(result_dir, args.exp_name+'.json')
        result_dict = []
        for i in range(0, pred_data.shape[0]):
            result_dict.append({'id': int(pred_id[i]), 'data': pred_data[i].tolist(), 'seg': pred_seg[i].tolist(), 'gt': gt_seg[i].tolist()})
        f = open(result_filepath, 'w')
        json.dump(result_dict, f, indent=4)
        f.close()

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f\n' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    for i in range(7):
        outstr += '%s : %.6f '%(part_colormap[i]['label'], test_parts_ious[i])                                                                       
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'pointnet', 'pointnet2', 'pointmanifold', 'pointmanifold2'],
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='pointchd', metavar='N',
                        choices=['pointchd'])
    parser.add_argument('--data_type', type=str, default='xyznxnynz', metavar='N',
                        choices=['xyz', 'xyznxnynz'],
                        help='The type of point cloud data')
    parser.add_argument('--data_norm', type=bool, default=False,
                        help='normalize the point cloud data to [-1, 1]')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=bool, default=False,
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='txt',
                        help='file format of visualization')
    parser.add_argument('--save_res', type=bool,  default=True,
                        help='save the test results')
    args = parser.parse_args()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
