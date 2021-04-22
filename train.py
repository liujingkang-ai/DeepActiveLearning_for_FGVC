# -*- coding: utf-8 -*-
import argparse
import time
import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import albumentations as A

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from models import net, resnest, resnet_origin
from utils.data_process import data_loader
from utils import losses
from utils.losses import metrics
from configs import config


cfg = config.Config()
if not cfg.use_distributed:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.single_device_id

if not os.path.exists(cfg.model_path):
    os.mkdir(cfg.model_path)

# whether use gpu
use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


if cfg.use_seed:
    # random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] =str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic =True


class Main():
    
    def data_process(self):
       return cfg.train_transform_t, cfg.val_transform_t


    def train_model(self, model, metric, epochs, train_loader, val_loader, cur_model_name, file_name):
        
        with open('./configs/config.py') as con_f:
            content = con_f.read()
        with open(os.path.join(cfg.model_path, file_name, cfg.log_file), 'a') as f:
            f.write( str(content) + '\n')

        if cfg.loss_func == 'label smooth':
            criterion = losses.labelsmooth.CrossEntropyLabelSmooth(cfg.num_class, epsilon=0.1)
        elif cfg.loss_func == 'focal loss':
            criterion = losses.focalloss.FocalLoss()
        else:
            criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss_weight))
        criterion.to(DEVICE)
        
        # if cfg.optim == 'adam':
        #     optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_rate)
        # elif cfg.optim == 'adamax':
        #     optimizer = optim.Adamax(model.parameters(), lr=cfg.lr, )
        # else:
        #     optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum_rate, weight_decay=cfg.weight_decay_rate, nesterov=cfg.use_nesterov)

        if cfg.optim == 'adam':
            optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay_rate)
        elif cfg.optim == 'adamax':
            optimizer = optim.Adamax([{'params': model.parameters()}, {'params': metric.parameters()}], lr=cfg.lr, )
        else:
            optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}], lr=cfg.lr, momentum=cfg.momentum_rate, weight_decay=cfg.weight_decay_rate, nesterov=cfg.use_nesterov)

        if cfg.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs/6), eta_min=1e-5)
        elif cfg.scheduler == 'cyclic':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
        elif cfg.scheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
             milestones=[int(epochs/3), int(2 * epochs / 3), int(4 * epochs / 5)], gamma=0.4)
        elif cfg.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/4), gamma=0.2)
        elif cfg.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        elif cfg.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        
        best_acc = 0.01
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            metric.train()
            train_loss = 0
            print('Current Model:', cur_model_name, '\tCurrent Learning Rate:', round(scheduler.get_last_lr()[0], 5))
            for i, (img, label) in enumerate(train_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                # print(img.size())
                # print(label.size())
                optimizer.zero_grad()
                # out, _ = model(img)
                feature = model(img)
                out = metric(feature, label)
                loss = criterion(out, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if i % cfg.print_iter == 0:
                    cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    info = (
                        f'Current Time: {cur_time}\tEpoch: {epoch + 1} / {epochs}, [{i} | {len(train_loader)}]'
                        f'\tTrain Loss: {train_loss:.4f}'
                    )
                    print(info)
                    train_loss = 0
                    with open(os.path.join(cfg.model_path, file_name, cfg.log_file), 'a') as f:
                        f.write(info + '\n')
            
            scheduler.step()
            if epoch % 1 == 0 or epoch == epochs - 1:
                correct = 0
                print('====== Evaluation ======')
                model.eval()
                metric.eval()
                with torch.no_grad():
                    for val_img, val_label in tqdm(val_loader):
                        val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)

                        # val_out, _ = model(val_img)
                        feature = model(val_img)
                        val_out = metric(feature, val_label)
                        val_pred = torch.max(val_out, 1)[1]
                        correct += torch.sum(val_pred == val_label).cpu().item()
                
                if correct / len(val_loader.dataset) >= best_acc:
                    best_acc = correct / len(val_loader.dataset)
                    best_epoch = epoch + 1
                info = (
                    f'Model: {cur_model_name}\tEpoch: {epoch + 1}\tAccuracy {100 * correct / len(val_loader.dataset):.4f}% [{correct} | {len(val_loader.dataset)}]'
                    f'\nBest_acc: {best_acc:.6f}, Best_epoch: {best_epoch}'
                )
                print(info)
                print()
                with open(os.path.join(cfg.model_path, file_name, cfg.log_file), 'a') as f:
                        f.write(info + '\n')
                
                cur_train_acc = round(correct / len(val_loader.dataset), 4)
                torch.save(model.state_dict(),
                 cfg.model_path + '/' + file_name + '/' + str(cur_model_name) +'_epoch_' + str(epoch + 1) + '_' + str(cur_train_acc) + '.pth')

        torch.save(model.state_dict(), cfg.model_path + '/' + file_name + '/'+ str(cur_model_name) + 'lastest.pth')
        print(f'Current model-{cur_model_name} ends, saved the model.')


    def train(self):

        df = pd.read_csv(os.path.join('../data/CUB200Birds/CUB_200_2011/', 'train.csv'))
        df= df.sample(frac=1.0)
        image_path_list = df['filename'].values
        label_list = df['value'].values

        # split dataset
        all_size = len(image_path_list)
        train_size = int(all_size * cfg.train_per)
        train_image_path_list = image_path_list[:train_size]
        train_label_list = label_list[:train_size]

        val_image_path_list = image_path_list[train_size:]
        val_label_list = label_list[train_size:]

        train_transform, val_transform = self.data_process()
        
        
        train_data = data_loader.ImageData(train_image_path_list, train_label_list, train_transform, cfg.train_data)
        val_data = data_loader.ImageData(val_image_path_list, val_label_list, val_transform, cfg.val_data)
        
        print('Trainset Size:', len(train_data))
        print('Valset Size:', len(val_data))
        
        train_loader = DataLoader(train_data, batch_size=cfg.train_batch_size, shuffle=True, num_workers = cfg.num_worker, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=cfg.val_batch_size, shuffle=False, num_workers = cfg.num_worker, drop_last=False)

        cur_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
        if not os.path.exists(os.path.join(cfg.model_path, cur_time)):
            os.mkdir(os.path.join(cfg.model_path, cur_time))
        file_name = cur_time

        # change
        model1 = resnet_origin.net('resnet50', cfg.pretrained, cfg.is_local, cfg.change_top, cfg.num_class*2)
        metric = metrics.ArcMarginProduct(cfg.num_class*2, cfg.num_class, s=30, m=0.5, easy_margin=False)
        # metric = metrics.AddMarginProduct(cfg.num_class*2, cfg.num_class, s=30, m=0.35)
        model1 = model1.to(DEVICE)
        metric = metric.to(DEVICE)
        if cfg.use_distributed:
            model1 = torch.nn.DataParallel(model1, device_ids=cfg.device_id)
            metric = torch.nn.DataParallel(metric, device_ids=cfg.device_id)
        if cfg.use_resume:
            model1.load_state_dict(torch.load(os.path.join(cfg.model_path, 'Tue Jun 30 08:21:26 2020', 'resnest101_epoch_26_0.8842.pth')))
        print('Loading model1...')
        self.train_model(model1, metric, cfg.epochs, train_loader, val_loader, 'resnet50_arc', file_name)

        # model2 = efficientnet.net('efficientnet-b3', cfg.pretrained, cfg.is_local, cfg.change_top, cfg.num_class)
        # model2 = model2.to(DEVICE)
        # if cfg.use_distributed:
        #     model2 = torch.nn.DataParallel(model2, device_ids=cfg.device_id)
        # if cfg.use_resume:
        #     model2.load_state_dict(torch.load(os.path.join(cfg.model_path, 'Tue Jun 30 22:13:40 2020', 'resnext101_epoch_14_0.8885.pth')))
        # print('Loading model2...')
        # self.train_model(model2, cfg.epochs, train_loader, val_loader, 'efficientnet-b3', file_name)

        # model3 = resnest.net('resnest101', cfg.pretrained, cfg.is_local, cfg.change_top, cfg.num_class)
        # model3 = model3.to(DEVICE)
        # if cfg.use_distributed:
        #     model3 = torch.nn.DataParallel(model3, device_ids=cfg.device_id)
        # if cfg.use_resume:
        #     model3.load_state_dict(torch.load(os.path.join(cfg.model_path, 'Fri Jul  3 09:33:45 2020', 'resnet101_epoch_16_0.8717.pth')))
        # print('Loading model3...')
        # self.train_model(model3, cfg.epochs, train_loader, val_loader, 'resnest101', file_name)

        # model4 = densenet.net('densenet169', cfg.pretrained, cfg.is_local, cfg.change_top, cfg.num_class)
        # model4 = model4.to(DEVICE)
        # if cfg.use_distributed:
        #     model4 = torch.nn.DataParallel(model4, device_ids=cfg.device_id)
        # if cfg.use_resume:
        #     model4.load_state_dict(torch.load(os.path.join(cfg.model_path, 'Fri Jul  3 09:33:45 2020', 'resnet101_epoch_16_0.8717.pth')))
        # print('Loading model4...')
        # self.train_model(model4, cfg.epochs, train_loader, val_loader, 'densenet169', file_name)
        
        # model1 = net.Net(num_class=cfg.num_class)
        # model1 = model1.to(DEVICE)
        # if cfg.use_distributed:
        #     model1 = torch.nn.DataParallel(model1, device_ids=cfg.device_id)
        # if cfg.use_resume:
        #     model1.load_state_dict(torch.load(os.path.join(cfg.model_path, 'Tue Jun 30 08:21:26 2020', 'net_epoch_26_0.8842.pth')))
        # print('Loading model1...')
        # self.train_model(model1, cfg.epochs, train_loader, val_loader, 'newnet', file_name)
        # metric = losses.arcface.ArcMarginProduct(256, cfg.num_class, s=30, m=0.5, easy_margin=False)
         
if __name__ == '__main__':
    main = Main()
    main.train()


