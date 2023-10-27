# -*- coding:utf-8 -*-

import torch
from dataset import load_data, generater, normalization, setup_seed

from train import *
from test import test
from module import applyPCA
import model
import os
from draw import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = 'Trento'

# dataset = 'MUUFL'
# dataset = 'Houston'
dataset = ['Trento', 'MUUFL', 'Houston']
batch_sizes = [16, 128, 32]
windowSize = [17, 13, 11]
num_classes = [6, 11, 15]
in_channels_2 = [1, 2, 1]
setup_seed(2021)
for i in range(3):

    HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset[i])

    # 归一化
    print(HSI_data.shape)
    print(LiDAR_data.shape)
    HSI_data, _ = applyPCA(HSI_data, numComponents=15)
    if i == 1:
        LiDAR_data, _ = applyPCA(LiDAR_data, numComponents=2)
        LiDAR_data = normalization(LiDAR_data, type=1)
    else:
        LiDAR_data, _ = applyPCA(LiDAR_data, numComponents=1)
        LiDAR_data = normalization(LiDAR_data, type=1)

    HSI_data = normalization(HSI_data, type=1)
    TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, all_iter, gt_iter = generater(HSI_data,
                                                                         LiDAR_data,
                                                                         Train_data,
                                                                         Test_data,
                                                                         GT,
                                                                         batch_size=batch_sizes[i],
                                                                         windowSize=windowSize[i]
                                                                         )



    print(max(train_iter.dataset.tensors[2])+1)
    '''
    model1 = train(dataset=dataset,
         train_iter=train_iter,
         device=device,
         epoches=320,
         ITER=1,
         TRAIN_SIZE=TRAIN_SIZE,
         TEST_SIZE=TEST_SIZE,
         TOTAL_SIZE=TOTAL_SIZE,
         test_iter=test_iter)
    '''
    '''
    model1 = train_speed(dataset=dataset,
         train_iter=train_iter,
         device=device,
         epoches=200,
         ITER=1,
         TRAIN_SIZE=TRAIN_SIZE,
         TEST_SIZE=TEST_SIZE,
         TOTAL_SIZE=TOTAL_SIZE,
         test_iter=test_iter)
    '''
    # model1 = train_best_model(dataset=dataset[i],
    #      train_iter=train_iter,
    #      device=device,
    #      epoches=200,
    #      ITER=1,
    #      TRAIN_SIZE=TRAIN_SIZE,
    #      TEST_SIZE=TEST_SIZE,
    #      TOTAL_SIZE=TOTAL_SIZE,
    #      test_iter=test_iter,
    #      num_classes=num_classes[i],
    #      in_channels_2=in_channels_2[i],
    #      windowSize=windowSize[i])

    model1 = fusion_main(15, in_channels_2[i], num_classes[i], windowSize[i]).to(device)
    #

    # classification, oa, aa, kappa, each_acc = test(test_iter=test_iter,
    #                                                device=device,
    #                                                dataset=dataset[i],
    #                                                net1=model1)
    # print(classification, oa, aa, kappa, each_acc)




    num = HSI_data.shape[0]*HSI_data.shape[1]
    # draw = position(num,
    #                     GT,
    #                     all_iter,
    #                     dataset[i],
    #                     device,
    #                     model1)

    # 真值
    index = 1
    draw = dis_groundtruth2(Train_data, dataset[i], index)
    index = 2
    draw = dis_groundtruth2(Test_data, dataset[i], index)
    index = 3
    draw = dis_groundtruth2(GT, dataset[i], index)

    # draw = position_gt(num,
    #                 GT,
    #                 gt_iter,
    #                 dataset[i],
    #                 device,
    #                 model1)

