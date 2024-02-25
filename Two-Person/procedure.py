from config import *
from model import *
from dataset import DataSet, Feeder_semi
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from einops import rearrange, repeat
from math import pi, cos

from module.gcn.st_gcn import Model
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
setup_seed(1)

class BaseProcessor:

    @ex.capture
    def load_data(self,train_list,train_label,test_list,test_label,batch_size,label_percent):
        self.dataset = dict()
        self.data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        # self.dataset['semi'] = Feeder_semi(train_list, train_label, label_percent)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=32,
            pin_memory=True,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            num_workers=32,
            pin_memory=True,
            shuffle=False)
        
        # self.data_loader['semi'] = torch.utils.data.DataLoader(
        #     dataset=self.dataset['semi'],
        #     batch_size=batch_size,
        #     num_workers=32,
        #     shuffle=True)
        
    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()
    
    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.test_epoch()
    
    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(self):
        
        pass

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

# %%
class RecognitionProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,dropout,
                    graph_args,edge_importance_weighting,people_importance_weighting,graph_args_1,weight_path):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                    hidden_dim=hidden_dim,dropout=dropout,
                                    graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    people_importance_weighting=people_importance_weighting,
                                    graph_args_1=graph_args_1,
                                    )
        self.encoder = self.encoder.cuda()
        self.classifier = BTwins_Linear().cuda()

        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, lp_lr, lp_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
             lr=lp_lr,
             )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, lp_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self, epoch, lp_epoch, lp_lr):
        self.encoder.eval()
        self.classifier.train()

        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data,'joint')
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):
        N = data.size(0)
        tensor = torch.zeros(N, 2, dtype=torch.int)
        for i in range(N):
            row = torch.arange(2, dtype=torch.int)
            tensor[i] = row
        Z,_ = self.encoder(data,[],tensor)
        Z = Z.detach()
        Z = Z.mean(dim=1)
        predict = self.classifier(Z,Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/train/cls_acc", acc.item())
        self.log.update_batch("log/train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, save_lp=True):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = './output/result_path/' + str(epoch) + '_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data,'joint')

            with torch.no_grad():
                N = data.size(0)
                tensor = torch.zeros(N, 2, dtype=torch.int)
                for i in range(N):
                    row = torch.arange(2, dtype=torch.int)
                    tensor[i] = row
                Z,_ = self.encoder(data, [], tensor)
                Z = Z.mean(dim=1)
                predict = self.classifier(Z,Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_lp:
            torch.save(result_list, r_path)
            torch.save(label_list, './output/result_path/' + str(epoch) + '_label.pkl')

    def save_model(self):
        
        pass
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class SemiProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, people_importance_weighting):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                             hidden_dim=hidden_dim, dropout=dropout,
                             graph_args=graph_args,
                             edge_importance_weighting=edge_importance_weighting,
                             people_importance_weighting=people_importance_weighting
                             )
        self.encoder = self.encoder.cuda()
        self.classifier = BTwins_Linear().cuda()
        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['semi']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data,'joint')
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label): 

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/semi_train/cls_acc", acc.item())
        self.log.update_batch("log/semi_train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_semi=True):
        self.encoder.eval()
        self.classifier.eval()

        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_semi10_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data,'joint')
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/semi_test/cls_acc", acc.item())
            self.log.update_batch("log/semi_test/cls_loss", loss.item())

        if save_semi:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class FTProcessor(BaseProcessor):
    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,dropout,
                    graph_args,edge_importance_weighting,people_importance_weighting,graph_args_1,weight_path):

        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                    hidden_dim=hidden_dim,dropout=dropout,
                                    graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    people_importance_weighting=people_importance_weighting,
                                    graph_args_1=graph_args_1,
                                    )
        self.encoder = self.encoder.cuda()
        self.classifier = BTwins_Linear().cuda()
        self.load_weights(self.encoder, weight_path)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        ft_epoch = 50
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch, eta_min=5e-7)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self,epoch):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data,'joint')
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        N = data.size(0)
        tensor = torch.zeros(N, 2, dtype=torch.int)
        for i in range(N):
            row = torch.arange(2, dtype=torch.int)
            tensor[i] = row
        Z,_ = self.encoder(data,[],tensor)
        Z = Z.mean(dim=1)
        predict = self.classifier(Z,Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/finetune/cls_acc", acc.item())
        self.log.update_batch("log/finetune/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, save_finetune=True):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = './output/result_path/' + str(epoch) + '_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data, 'joint')

            with torch.no_grad():
                N = data.size(0)
                tensor = torch.zeros(N, 2, dtype=torch.int)
                for i in range(N):
                    row = torch.arange(2, dtype=torch.int)
                    tensor[i] = row
                Z,_ = self.encoder(data, [], tensor)
                Z = Z.mean(dim=1)
                predict = self.classifier(Z, Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_finetune:
            torch.save(result_list, r_path)
            torch.save(label_list, './output/result_path/' + str(epoch) + '_label.pkl')


    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)


def rand_bbox(size, lam, mix):
    T = size[2]
    W = size[3]
    H = size[4]

    if mix in ['cutmix', 'cutmixup']:
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = 0
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = T
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    elif mix in ['framemix', 'framemixup']:
        cut_rat = 1. - lam
        cut_t = np.int(T * cut_rat)

        ct = np.random.randint(T)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = 0
        bby1 = 0
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = W
        bby2 = H
    else: # spatio-temporal, cubemix
        cut_rat = np.power(1. - lam, 1./3.)
        cut_t = int(T * cut_rat)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbt1, bbx1, bby1, bbt2, bbx2, bby2

def local_skeleton_people_temporal_mix_regularization(inputs_1, inputs2_1, mix_type, beta):
    if mix_type in ['mixup', 'cutmix', 'framemix', 'cubemix', 'framemixup', 'fademixup', 'cutmixup', 'cubemixup']:
        # Sample Mix Ratio (Lambda)
        f = np.random.rand(1)
        if f < 0.33:
            ls = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            inputs = inputs_1[:, :, :, ls, :]
            inputs2 = inputs2_1[:, :, :, ls, :]
        elif f > 0.66:
            ls = [0, 3, 6, 9]
            inputs = inputs_1[:, :, ls, :, :]
            inputs2 = inputs2_1[:, :, ls, :, :]
        else:
            ls = [2, 3, 4]
            inputs = inputs_1[:, :, :, :, ls]
            inputs2 = inputs2_1[:, :, :, :, ls]

        lam = np.random.beta(beta, beta)
        lamt = lam

        # Random Mix within Batch
        rand_index = torch.randperm(inputs.size()[0]).cuda()


        if mix_type in ['cutmix', 'framemix', 'cubemix', 'cutmixup', 'framemixup', 'cubemixup']:
            # Sample Mixing Coordinates
            bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(inputs.size(), lam, mix_type)
            lamt = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1)/ (inputs.size()[-1] * inputs2.size()[-2] * inputs2.size()[-3]))

            if mix_type in ['cutmixup', 'framemixup', 'cubemixup']:

                mix_tmp = inputs * lamt + inputs2 * (1. - lamt)
                fr = np.random.rand(1)
                if fr < 0.5:  # Basic MixUp, 0.5 Prob FrameMixUp
                    if lamt >= 0.5:
                        # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                         bby1:bby2]

                    else:
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]

                inputs = mix_tmp
            else:
                # Mix
                inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                bby1:bby2]
        else:  # mixup: blending two videos
            # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
            if mix_type in ['mixup']:
                inputs = inputs * lam + inputs2 * (1. - lam)
            elif mix_type in ['fademixup']:  # temporal-mix-up
                adj = np.random.choice([-1, 1]) * np.random.uniform(0, min(lam, 1.0 - lam))
                fade = np.linspace(lam - adj, lam + adj, num=inputs.size(2))
                for taxis in range(inputs.size(2)):
                    inputs[:, :, taxis, :, :] = inputs[:, :, taxis, :, :] * fade[taxis] + inputs[rand_index, :, taxis,
                                                                                  :, :] * (1. - fade[taxis])
        if f < 0.33:
            inputs_1[:, :, :, ls, :] = inputs
        elif f > 0.66:
            inputs_1[:, :, ls, :, :] = inputs
        else:
            inputs_1[:, :, :, :, ls] = inputs



    return inputs_1,lamt

def local_skeleton_people_mix_regularization(inputs_1, inputs2_1, mix_type, beta):
    if mix_type in ['mixup', 'cutmix', 'framemix', 'cubemix', 'framemixup', 'fademixup', 'cutmixup', 'cubemixup']:
        # Sample Mix Ratio (Lambda)
        f = np.random.rand(1)
        if f < 0.33:
            ls = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            inputs = inputs_1[:, :, :, ls, :]
            inputs2 = inputs2_1[:, :, :, ls, :]
        elif f > 0.66:
            inputs = inputs_1
            inputs2 = inputs2_1
        else:
            ls = [2, 3, 4]
            inputs = inputs_1[:, :, :, :, ls]
            inputs2 = inputs2_1[:, :, :, :, ls]

        lam = np.random.beta(beta, beta)
        lamt = lam

        # Random Mix within Batch
        rand_index = torch.randperm(inputs.size()[0]).cuda()


        if mix_type in ['cutmix', 'framemix', 'cubemix', 'cutmixup', 'framemixup', 'cubemixup']:
            # Sample Mixing Coordinates
            bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(inputs.size(), lam, mix_type)
            lamt = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1)/ (inputs.size()[-1] * inputs2.size()[-2] * inputs2.size()[-3]))

            if mix_type in ['cutmixup', 'framemixup', 'cubemixup']:

                mix_tmp = inputs * lamt + inputs2 * (1. - lamt)
                fr = np.random.rand(1)
                if fr < 0.5:  # Basic MixUp, 0.5 Prob FrameMixUp
                    if lamt >= 0.5:
                        # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                         bby1:bby2]

                    else:
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]

                inputs = mix_tmp
            else:
                # Mix
                inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                bby1:bby2]
        else:  # mixup: blending two videos
            # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
            if mix_type in ['mixup']:
                inputs = inputs * lam + inputs2 * (1. - lam)
            elif mix_type in ['fademixup']:  # temporal-mix-up
                adj = np.random.choice([-1, 1]) * np.random.uniform(0, min(lam, 1.0 - lam))
                fade = np.linspace(lam - adj, lam + adj, num=inputs.size(2))
                for taxis in range(inputs.size(2)):
                    inputs[:, :, taxis, :, :] = inputs[:, :, taxis, :, :] * fade[taxis] + inputs[rand_index, :, taxis,
                                                                                  :, :] * (1. - fade[taxis])
        if f < 0.33:
            inputs_1[:, :, :, ls, :] = inputs
        elif f > 0.66:
            inputs_1 = inputs
        else:
            inputs_1[:, :, :, :, ls] = inputs



    return inputs_1,lamt

def sign(x):
    return torch.sign(x.sign() + 0.5)

def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)

def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)

def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
            - 1 / 11 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x**3
    elif order == 2:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5
    elif order == 3:
        return (
            x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5 - 1 / 7 * k**3 * x**7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k**2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(1e-15)


def dist(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, keepdim=False, dim=-1):
    r"""
    Compute the geodesic distance between :math:`x` and :math:`y` on the manifold.

    .. math::

        d_\kappa(x, y) = 2\tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)

    .. plot:: plots/extended/stereographic/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`

    """



    return _dist(x, y, w, keepdim=keepdim, dim=dim)








def _dist(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )


def mix_regularization(inputs, inputs2, mix_type, beta):
    if mix_type in ['mixup', 'cutmix', 'framemix', 'cubemix', 'framemixup', 'fademixup', 'cutmixup', 'cubemixup']:
        # Sample Mix Ratio (Lambda)
        lam = np.random.beta(beta, beta)

        lamt = lam
        # Random Mix within Batch
        rand_index = torch.randperm(inputs.size()[0]).cuda()


        if mix_type in ['cutmix', 'framemix', 'cubemix', 'cutmixup', 'framemixup', 'cubemixup']:
            # Sample Mixing Coordinates
            bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(inputs.size(), lam, mix_type)
            lamt = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1)/ (inputs.size()[-1] * inputs2.size()[-2] * inputs2.size()[-3]))

            if mix_type in ['cutmixup', 'framemixup', 'cubemixup']:

                mix_tmp = inputs * lamt + inputs2 * (1. - lamt)
                fr = np.random.rand(1)
                if fr < 0.5:  # Basic MixUp, 0.5 Prob FrameMixUp
                    mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]

                inputs = mix_tmp
            else:
                # Mix
                inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                bby1:bby2]
        else:  # mixup: blending two videos
            # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
            if mix_type in ['mixup']:
                inputs = inputs * lam + inputs2 * (1. - lam)
            elif mix_type in ['fademixup']:  # temporal-mix-up
                adj = np.random.choice([-1, 1]) * np.random.uniform(0, min(lam, 1.0 - lam))
                fade = np.linspace(lam - adj, lam + adj, num=inputs.size(2))
                for taxis in range(inputs.size(2)):
                    inputs[:, :, taxis, :, :] = inputs[:, :, taxis, :, :] * fade[taxis] + inputs[rand_index, :, taxis,
                                                                                          :, :] * (1. - fade[taxis])
    return inputs,lamt

def local_skeleton_people_mixup(inputs_1, inputs2_1, mix_type, beta):

    if mix_type in ['mixup', 'cutmix', 'framemix', 'cubemix', 'framemixup', 'fademixup', 'cutmixup', 'cubemixup']:
        shear = inputs_1.clone()
        crop = inputs2_1.clone()
        lamt = np.random.beta(beta, beta)
        ls_p = [4,5,6,7]
        ls_s = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        inputs_p = inputs_1[:, :, :, :, ls_p]
        inputs2_p = inputs2_1[:, :, :, :, ls_p]
        inputs = inputs_p[:, :, :, ls_s, :]
        inputs2 = inputs2_p[:, :, :, ls_s, :]
        mix_tmp = inputs * lamt + inputs2 * (1. - lamt)
        inputs_1[:, :, :, ls_s, 2] = mix_tmp[...,0]
        inputs_1[:, :, :, ls_s, 3] = mix_tmp[...,1]


    return inputs_1,lamt,shear,crop

def global_mix_regularization_lambdasame(inputs_1, inputs2_1, mix_type, beta):
    shear = inputs_1.clone()
    crop = inputs2_1.clone()
    out = shear.clone()
    out2 = crop.clone()
    if mix_type in ['cubemixup']:

        lam = np.random.beta(beta, beta)

        bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(shear.size(), lam, mix_type)

        lamt = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1) / (
                    shear.size()[-1] * shear.size()[-2] * shear.size()[-3]))

        mix_tmp = shear * lam + crop * (1. - lam)

        if lam <= 0.5:
            # TODO: 这里的 rand-index 会把之前的顺序给打乱，mix的数据可能就不是原来的数据了，性能可能会下降，跑的时候要做好记录
            out[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
            return out, lamt, shear, crop, lam

        else:
            out2[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
            return out2, lamt, shear, crop, lam

class BTProcessor(BaseProcessor):
    
    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,dropout,
                    graph_args,edge_importance_weighting,people_importance_weighting,graph_args_1):

        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                    hidden_dim=hidden_dim,dropout=dropout,
                                    graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    people_importance_weighting=people_importance_weighting,
                                    graph_args_1=graph_args_1,
                                    )
        self.encoder = self.encoder.cuda()
        self.btwins_head = BTwins().cuda()
        self.transmodel = VideoNet(512, 8, 6, 1250, 1024, 0.1).cuda()

    @ex.capture
    def load_optim(self, pretrain_lr, pretrain_epoch, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.btwins_head.parameters()},
            ], 
            weight_decay=weight_decay,
            lr=pretrain_lr)

    def btwins_batch(self, feat1, feat2, mode):
        BTloss = self.btwins_head(feat1, feat2)
        BTloss = torch.mean(BTloss)

        return BTloss

    @ex.capture
    def train_epoch(self, epoch, pretrain_epoch, pretrain_lr):
        self.encoder.train()
        self.btwins_head.train()
        self.transmodel.train()

        loader = self.data_loader['train']
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=pretrain_epoch,
                                  lr_min=pretrain_lr * 0.0001, lr_max=pretrain_lr)

        for data, label in tqdm(loader):
            # load data
            data = data.type(torch.FloatTensor).cuda()
            data = get_stream(data, 'joint')
            N = data.size(0)
            tensor = torch.zeros(N, 2, dtype=torch.int)
            for i in range(N):
                row = torch.arange(2, dtype=torch.int)
                tensor[i] = row

            input2 = data
            input2 = motion_att_temp_global_small_mask_score(input2)

            feat2, feat2_H = self.encoder(input2, [], tensor)
            feat2_mean = feat2.mean(dim=1)
            feat2_H_mean = feat2_H.mean(dim=1)
            # loss = self.btwins_batch(feat2_mean, feat2_H_mean, mode='temp_mask')
            loss = 0

            # mask_shape = feat2.shape
            # alpha = 0.6
            # mask = np.random.beta(alpha, alpha, size=mask_shape)
            # if isinstance(mask, np.ndarray):
            #     mask = torch.from_numpy(mask).float().cuda()
            #
            #
            # # TODO: 1.这个位置到底放哪里？应该是取了数据以后再取平均还是取之前取平均
            # # TODO: 2.feat3需不需要进行 mixup
            # feat2 = mask * feat2 + (1 - mask) * feat3
            # # feat3 = mask * feat3 + (1 - mask) * feat2

            input3 = data
            # input3,ignore_people = people_att_temp_small_mask(input3)
            # feat3 = self.encoder(input3,[],ignore_people)
            N, C, T, V, M = input3.size()
            input3_1 = crop(data)
            input3_2 = shear(data)
            input3, lamt, input3_crop, input3_shear, lam = global_mix_regularization_lambdasame(input3_1, input3_2, 'cubemixup', 1.2)
            feat3, feat3_H = self.encoder(input3, [], tensor)
            feat3_mean = feat3.mean(dim=1)
            feat3_H_mean = feat3_H.mean(dim=1)
            # loss = loss + self.btwins_batch(feat3_mean, feat3_H_mean, mode='temp_mask')

            encoded_feat3 = feat3.view(N, M, -1)
            encoded_feat2 = feat2.view(N, M, -1)

            encoded_feat3_1, _ = self.encoder(input3_crop, [], tensor)
            encoded_feat3_2, _ = self.encoder(input3_shear, [], tensor)

            people_list, feat3 = self.transmodel(encoded_feat3)
            _, feat2 = self.transmodel(encoded_feat2)

            people_list1, feat3_1 = self.transmodel(encoded_feat3_1)
            people_list2, feat3_2 = self.transmodel(encoded_feat3_2)
            people_list = repeat(people_list, 'n m -> n m c', c=512)


            mean_feat2 = feat2.mean(dim=1)
            mean_feat3 = feat3.mean(dim=1)


            mean_encoded_feat2 = encoded_feat2.mean(dim=1)
            mean_encoded_feat3 = encoded_feat3.mean(dim=1)
            #
            # perm = torch.randperm(subgroup_feat3.size(0)).cuda()
            # subgroup_feat3_shuffled = subgroup_feat3.index_select(0, perm)
            # cos_sim = torch.nn.functional.cosine_similarity(subgroup_feat3, subgroup_feat3_shuffled, dim=1)
            # remain_cos_sim = 1 - cos_sim
            # remain_cos_sim = remain_cos_sim.unsqueeze(1)
            # cos_sim = cos_sim.unsqueeze(1)
            # mix_cos_subgroup_feat3 = cos_sim * subgroup_feat3_shuffled + remain_cos_sim * subgroup_feat3
            # mix_cos_subgroup_feat3_mean = mix_cos_subgroup_feat3.mean(dim=1)
            # # print(subgroup_mean_feat2.shape, mix_cos_subgroup_feat3.shape)
            # loss = loss + self.btwins_batch(subgroup_mean_encoded_feat2, mix_cos_subgroup_feat3_mean, mode='temp_mask')
            perm = torch.randperm(feat3.size(0)).cuda()
            feat3_shuffled = feat3.index_select(0, perm)
            hyperbolic = dist(feat3, feat3_shuffled, w=torch.tensor(1.), dim=1)

            remain_hyperbolic = 1 - hyperbolic
            remain_hyperbolic = remain_hyperbolic.unsqueeze(1)
            hyperbolic = hyperbolic.unsqueeze(1)
            mix_cos_feat3 = hyperbolic * feat3_shuffled + remain_hyperbolic * feat3
            mix_cos_feat3_mean = mix_cos_feat3.mean(dim=1)
            loss = loss + self.btwins_batch(mean_feat2, mix_cos_feat3_mean, mode='temp_mask')

            # subgroup_mean_feat3_1 = subgroup_feat3_1.mean(dim=1)
            # subgroup_mean_feat3_2 = subgroup_feat3_2.mean(dim=1)
            #
            # group_mean_feat3 = feat3.mean(dim=1)
            # group_mean_feat2 = feat2.mean(dim=1)
            #
            # group_mean_feat3_1 = feat3_1.view(N, M, -1).mean(dim=1)
            # group_mean_feat3_2 = feat3_2.view(N, M, -1).mean(dim=1)
            # loss_3_1 = self.btwins_batch(group_mean_feat3, group_mean_feat3_1, mode='temp_mask')
            # loss_3_2 = self.btwins_batch(group_mean_feat3, group_mean_feat3_2, mode='temp_mask')
            # loss_ss_3_1 = self.btwins_batch(subgroup_mean_feat3, subgroup_mean_feat3_1, mode='temp_mask')
            # loss_ss_3_2 = self.btwins_batch(subgroup_mean_feat3, subgroup_mean_feat3_2, mode='temp_mask')
            # loss_gs_3_1 = self.btwins_batch(group_mean_feat3, subgroup_mean_feat3_1, mode='temp_mask')
            # loss_gs_3_2 = self.btwins_batch(group_mean_feat3, subgroup_mean_feat3_2, mode='temp_mask')

            # loss3 = lamt * loss_3_1 + (1 - lamt) * loss_3_2
            # loss_gs_3 = lamt * loss_gs_3_1 + (1 - lamt) * loss_gs_3_2
            # loss_ss_3 = lamt * loss_ss_3_1 + (1 - lamt) * loss_ss_3_2

            # loss_gg = self.btwins_batch(group_mean_feat3, group_mean_feat2, mode='temp_mask')
            # loss_gs1 = self.btwins_batch(group_mean_feat3, subgroup_mean_feat2, mode='temp_mask')
            # loss_gs2 = self.btwins_batch(subgroup_mean_feat3, group_mean_feat2, mode='temp_mask')
            # loss_ss = self.btwins_batch(subgroup_mean_feat3, subgroup_mean_feat2, mode='temp_mask')
            # loss = loss_ss + loss_gs2 + loss_gs1 + loss_gg + loss3 + loss_gs_3 + loss_ss_3

            for i in range(feat2.size(1)):

                single_feat2 = feat2[:, i, :]
                encoded_single_feat2 = encoded_feat2[:, i, :]
                single_feat3_1 = feat3_1[:, i, :]
                single_feat3_2 = feat3_2[:, i, :]
                encoded_single_feat3_2 = encoded_feat3_2[:, i, :]
                encoded_single_feat3_1 = encoded_feat3_1[:, i, :]
                # encoded_single_feat3 = subgroup_encoded_feat3[:,i,:]
                loss_encoded_ps_3_1 = self.btwins_batch(encoded_single_feat3_1, mean_encoded_feat3,
                                                        mode='temp_mask')
                loss_encoded_ps_3_2 = self.btwins_batch(encoded_single_feat3_2, mean_encoded_feat3,
                                                        mode='temp_mask')
                if lam < 0.5:
                    loss_encoded_ps_3 = loss_encoded_ps_3_1 * (lam * lamt + (1 - lam)) + loss_encoded_ps_3_2 * ((1 - lam) * lamt)
                else:
                    loss_encoded_ps_3 = loss_encoded_ps_3_1 * (lam * lamt) + loss_encoded_ps_3_2 * (1 - lam * lamt)

                # loss_encoded_pp = self.btwins_batch(single_feat2, encoded_single_feat3, mode='temp_mask')
                loss_encoded_ps1 = self.btwins_batch(encoded_single_feat2, mean_encoded_feat3,
                                                     mode='temp_mask')
                # loss_pp3_1 = self.btwins_batch(single_feat3_1, single_feat3, mode='temp_mask')
                # loss_pp3_2 = self.btwins_batch(single_feat3_2, single_feat3, mode='temp_mask')
                # loss_pp_3 = loss_pp3_1 * lamt + loss_pp3_2 * (1 - lamt)
                # loss_pg_3_1 = self.btwins_batch(single_feat3_1, group_mean_feat3, mode='temp_mask')
                # loss_pg_3_2 = self.btwins_batch(single_feat3_2, group_mean_feat3, mode='temp_mask')
                # loss_pg_3 = loss_pg_3_1 * lamt + loss_pg_3_2 * (1 - lamt)
                loss_ps_3_1 = self.btwins_batch(single_feat3_1, mean_feat3, mode='temp_mask')
                loss_ps_3_2 = self.btwins_batch(single_feat3_2, mean_feat3, mode='temp_mask')
                if lam < 0.5:
                    loss_ps_3 = loss_ps_3_1 * (lam * lamt + (1 - lam)) + loss_ps_3_2 * ((1 - lam) * lamt)
                else:
                    loss_ps_3 = loss_ps_3_1 * (lam * lamt) + loss_ps_3_2 * (1 - lam * lamt)
                # loss_ps_3 = loss_ps_3_1 * lamt + loss_ps_3_2 * (1 - lamt)
                # loss_pp = self.btwins_batch(single_feat2, single_feat3, mode='temp_mask')
                # loss_pp_1 = self.btwins_batch(single_feat2, single_feat3_1, mode='temp_mask')
                # loss_pp_2 = self.btwins_batch(single_feat2, single_feat3_2, mode='temp_mask')
                # loss_pp_3 = loss_pp_1 * lamt + loss_pp_2 * (1 - lamt)
                loss_ps1 = self.btwins_batch(single_feat2, mean_feat3, mode='temp_mask')
                # loss_ps2 = self.btwins_batch(subgroup_mean_feat2, single_feat3, mode='temp_mask')
                # loss_pg1 = self.btwins_batch(group_mean_feat2, single_feat3, mode='temp_mask')
                # loss_pg2 = self.btwins_batch(single_feat2, group_mean_feat3, mode='temp_mask')
                loss = loss_ps1 + loss + loss_ps_3 + loss_encoded_ps_3 + loss_encoded_ps1
                if i == 3:
                    break

            self.log.update_batch("log/pretrain/" + 'temp_mask' + "_bt_loss", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
  
    @ex.capture
    def save_model(self,epoch,version):
        torch.save(self.encoder.state_dict(), f"output/multi_model/xsub/v"+version+"_epoch_"+str(epoch+1)+"_pretrain.pt")
        
    @ex.capture
    def optimize(self, pretrain_epoch):
        for epoch in range(pretrain_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
            self.train_epoch(epoch=epoch)
            if epoch+1 == pretrain_epoch or (epoch+1)% 300 == 0 or epoch == 0:
                self.save_model(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,pretrain_epoch,lr=lr)
            
    @ex.capture
    def start(self):
        self.initialize()
        self.optimize()


# %%
@ex.automain
def main(train_mode):
    if "pretrain" in train_mode:
        p = BTProcessor()
    elif "lp" in train_mode:
        p = RecognitionProcessor()
    elif "ft" in train_mode:
        p = FTProcessor()
    elif "semi" in train_mode:
        p = SemiProcessor()
    else:
        print('train_mode error')
    p.start()