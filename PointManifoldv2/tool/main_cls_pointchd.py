#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM

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
from model.main_models import PointManifold_NNML, PointManifold2
import numpy as np
from torch.utils.data import DataLoader
from util.util import cls_loss, IOStream
import sklearn.metrics as metrics
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cls_label_dict = {0: 'ASD', 1: 'VSD', 2: 'AVSD', 3: 'ToF', 4: 'TGA', 5: 'DORV', 6: 'CAT', 7: 'CA', 8: 'AAH',
 9: 'DAA', 10: 'IAA', 11: 'PuA', 12: 'APVC', 13: 'DSVC', 14: 'PDA', 15: 'PAS', 16: 'Normal'}


def draw_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)


def train(args, io):
    train_loader = DataLoader(PointCHD(partition='train', num_points=args.num_points, task='cls', data_type=args.data_type, norm=args.data_norm), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCHD(partition='val', num_points=args.num_points, task='cls', data_type=args.data_type, norm=args.data_norm), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointmanifold':
        model = PointManifold_NNML(args, output_channels=17).to(device)
    elif args.model == 'pointmanifold2':
        model = PointManifold2(args, output_channels=17).to(device)
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
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = cls_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label, ids in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            logits = 1 / (1+torch.exp(-logits)) # sigmoid
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            pred_thres = args.pred_th
            preds = (logits > pred_thres).float()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch,
                                                            train_loss*1.0/count,
                                                            metrics.accuracy_score(
                                                            train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label, ids in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            logits = 1 / (1+torch.exp(-logits))
            if data.shape[0] == 1:
                label = label.unsqueeze(0) 
            loss = criterion(logits, label)
            pred_thres = args.pred_th
            preds = (logits > pred_thres).float()
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch,
                                                        test_loss*1.0/count,
                                                        test_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(PointCHD(partition='val', num_points=args.num_points, task='cls', data_type=args.data_type, norm=args.data_norm, seg_result=args.seg_result),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, output_channels=17).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2(args, output_channels=17).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args, output_channels=17).to(device)
    elif args.model == 'pointmanifold':
        model = PointManifold_NNML(args, output_channels=17).to(device)
    elif args.model == 'pointmanifold2':
        model = PointManifold2(args, output_channels=17).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label, ids in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        logits = 1 / (1+torch.exp(-logits))
        pred_thres = args.pred_th
        # for i in range(0, logits.shape[0]):
        #     print(logits[i])
        if data.shape[0] == 1:
            label = label.unsqueeze(0) 
        preds = (logits > pred_thres).float()
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        if args.save_res:
            preds_np = preds.cpu().numpy()
            ids_np = ids.cpu().numpy()
            gts_np = label.cpu().numpy()
            result_dir = os.path.join('outputs', args.exp_name, 'result')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            for i in range(0, data.shape[0]):
                pred = preds_np[i].tolist()
                gt = gts_np[i].tolist()
                id = ids_np[i] 
                pred_label = []
                gt_label = []
                for i, idx in enumerate(pred):
                    if idx == 1:
                        pred_label.append(cls_label_dict[i])
                for i, idx in enumerate(gt):
                    if idx == 1:
                        gt_label.append(cls_label_dict[i])
                filepath = os.path.join(result_dir, str(id)+'.json')
                result_dict = {'pred': pred,
                                'gt':gt,
                                'pred_label': pred_label,
                                'gt_label': gt_label}
                f = open(filepath, 'w')
                f.write(json.dumps(result_dict))
                f.close()

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_confusion_matrix = metrics.multilabel_confusion_matrix(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, class-wise results:\n'%(test_acc)
    for i in range(0, test_confusion_matrix.shape[0]):
        tn, fp, fn, tp = test_confusion_matrix[i].ravel()
        acc = tp / (tp+fn)
        outstr += str(cls_label_dict[i])+': %.6f'%(acc) + '\n'
    io.cprint(outstr)
    if args.save_res:
        result_dir = os.path.join('outputs', args.exp_name, 'result')
        labels = [cls_label_dict[i] for i in range(0, 17)]
        fig, ax = plt.subplots(5, 4, figsize=(12, 8))
        for i, (axes, cfs_matrix, label) in enumerate(zip(ax.flatten(), test_confusion_matrix, labels)):
            if i <= 16:
                draw_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
        
        fig.tight_layout()
        plt.savefig(os.path.join(result_dir, 'confusion_matrixes.png'), dpi=600, format='png')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
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
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--pred_th', type=float, default=0.9, metavar='pred_th',
                        help='Threshold for the predcition')                        
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_res', type=bool,  default=True,
                        help='save the test results')
    parser.add_argument('--seg_result', type=str,  default='',
                        help='The path of part segmentation result')
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
