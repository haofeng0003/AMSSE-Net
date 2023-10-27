import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import io
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import time

def colormap(dataset):
    if dataset == 'Trento':
        cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict,N=6)

    if dataset == 'MUUFL':
        cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict, N=11)

    if dataset == 'Houston':
        cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict, N=15)


def dis_groundtruth(gt, dataset):
    plt.imshow(gt, cmap=colormap(dataset))
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])

    plt.savefig(dataset+'.png', dpi=1200, format='png')
    plt.show()


def position(TOTAL_SIZE1, Y, all_iter, dataset, device, net1):
    all_num_part = TOTAL_SIZE1
    all_pixels = np.where(Y != -1)
    all_category = np.zeros([all_num_part, 1])
    result_pic = np.zeros_like(Y)
    net1.load_state_dict(torch.load('./models/' + dataset + '.pt'))  # 加载保存好的模型

    print('\n***Start  DRAw***\n')
    tick1 = time.time()
    y_test = []
    y_pred = []
    test_acc_sum = 0.0

    with torch.no_grad():
        for step, (X_hsi, X_lidar) in enumerate(all_iter):
            net1.eval()
            X_hsi = X_hsi.to(device)
            X_lidar = X_lidar.to(device)
            # y = y.to(device)
            out, out0, out1, out2, beta = net1(X_hsi, X_lidar)
            net1.train()
            # test_acc_sum += (out.argmax(dim=-1) == y.to(device)).float().sum().cpu().item()

            y_pred.append(out.argmax(dim=-1).cpu().numpy())
    # y_pred = y_pred.numpy()
    y_pred = [i for j in range(len(y_pred)) for i in y_pred[j]]
    # all_category = torch.from_numpy(all_category).long()

    all_category[:, 0] = y_pred[:]  # 将预测结果与像素点一一对应

    for k in range(all_num_part):
        row = all_pixels[0][k]  # 得到像素点标签的横坐标
        col = all_pixels[1][k]
        result_pic[row, col] = all_category[k, 0] + 1
    dis_groundtruth(result_pic, dataset)

    return net1


def position_gt(TOTAL_SIZE1, Y, all_iter, dataset, device, net1):
    all_num_part = TOTAL_SIZE1
    all_pixels = np.where(Y != 0)
    all_category = np.zeros([all_pixels[0].size, 1])
    result_pic = np.zeros_like(Y)
    net1.load_state_dict(torch.load('./models/' + dataset + '.pt'))  # 加载保存好的模型

    print('\n***Start  DRAw***\n')
    tick1 = time.time()
    y_test = []
    y_pred = []
    test_acc_sum = 0.0

    with torch.no_grad():
        for step, (X_hsi, X_lidar, y) in enumerate(all_iter):
            net1.eval()
            X_hsi = X_hsi.to(device)
            X_lidar = X_lidar.to(device)
            # y = y.to(device)
            out, out0, out1, out2, beta = net1(X_hsi, X_lidar)
            net1.train()
            test_acc_sum += (out.argmax(dim=-1) == y.to(device)).float().sum().cpu().item()

            y_pred.append(out.argmax(dim=-1).cpu().numpy())
    # y_pred = y_pred.numpy()
    print(test_acc_sum)
    y_pred = [i for j in range(len(y_pred)) for i in y_pred[j]]
    # all_category = torch.from_numpy(all_category).long()

    all_category[:, 0] = y_pred[:]  # 将预测结果与像素点一一对应

    for k in range(all_pixels[0].size):
        row = all_pixels[0][k]  # 得到像素点标签的横坐标
        col = all_pixels[1][k]
        result_pic[row, col] = all_category[k, 0]+1


    dis_groundtruth1(result_pic, dataset)

    return net1


def dis_groundtruth1(gt, dataset):
    plt.imshow(gt, cmap=colormap1(dataset))
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])

    plt.savefig(dataset+'1.png', dpi=1200, format='png')
    plt.show()


# 真值
def dis_groundtruth2(gt, dataset, index):
    plt.imshow(gt, cmap=colormap1(dataset))
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])

    plt.savefig(dataset+'-'+str(index)+'.png', dpi=1200, format='png')
    plt.show()

def colormap1(dataset):
    if dataset == 'Trento':
        cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict, N=7)

    if dataset == 'MUUFL':
        cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict, N=12)

    if dataset == 'Houston':
        cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']

        return colors.ListedColormap(cdict, N=16)

