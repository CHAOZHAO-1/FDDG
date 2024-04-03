import sys
import torch
import numpy as np
from torch import optim

from utilss import LabelSmoothingLoss, GradualWarmupScheduler
import  time
from torch.autograd import Variable
import  Coral
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class localTrain_1(object):
    def __init__(self, fetExtrac, classifier, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()
        self.classifier = classifier.cuda()

        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier.parameters()), args.lr0,  weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_task)


    def train(self):



        ac = [0.]
        best_model_dict = {}



        # local training (E1)
        for i in range(self.args.epochs):

            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch) ##7s
        # #
            # update learning rate


            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])

        #
            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()


        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return  loc_w,best_model_dict['C']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()


        self.fetExtrac.train()
        self.classifier.train()

        self.opti_task.zero_grad()

        feature = self.fetExtrac(x)
        pre = self.classifier(feature)
        loss_cla = self.lossFunc(pre, y)


        loss_cla.backward()
        self.opti_task.step()



        return loss_cla.item()


class localTrain_2(object):
    def __init__(self, fetExtrac,  generator, classifier, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()

        self.generator = generator.cuda()

        self.classifier = classifier.cuda()

        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()), args.lr0,  weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_task)
        afsche_gen = optim.lr_scheduler.ReduceLROnPlateau(self.generator.optimizer, factor=args.factor,
                                                          patience=args.patience,threshold=args.lr_threshold,
                                                          min_lr=1e-7)
        self.sche_gene = GradualWarmupScheduler(self.generator.optimizer, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_gen)


    def train(self):

        ac = [0.]

        best_model_dict = {}

        # local training (E1)
        for i in range(self.args.epochs):
            copy_loss = 0

            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                copy_loss+=self.train_step(batch) ##7s
            # print(copy_loss)

        # #
            # update learning rate

            self.sche_gene.step(i+1, 1. - ac[-1])
            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])

            best_model_dict['F'] = self.fetExtrac.state_dict()
            best_model_dict['G'] = self.generator.state_dict()



        return best_model_dict['G']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()
        randomn = torch.rand(y.size(0), self.args.input_size).cuda()


        # # training discriminator
        self.fetExtrac.eval()
        #
        y_onehot = torch.zeros(y.size(0), self.args.class_num).cuda()
        y_onehot.scatter_(1, y.view(-1, 1), 0.7).cuda()


        # training distribution generator
        self.generator.train()
        self.generator.optimizer.zero_grad()

        realz = self.generator(y_onehot,randomn)

        fakez = self.fetExtrac(x)


        pre = self.classifier(realz)
        loss_cla = self.lossFunc(pre, y)


        # loss_gene = Coral.CORAL(fakez, realz)
        loss_gene = Coral.mmd_rbf_noaccelerate(fakez, realz)+loss_cla



        loss_gene.backward()
        self.generator.optimizer.step()


        return loss_gene.item()

class localTrain_4(object):
    def __init__(self, fetExtrac, generator1, generator2, classifier, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()
        self.classifier = classifier.cuda()

        self.generator1 = generator1.cuda()
        self.generator2 = generator2.cuda()


        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier.parameters()), args.lr0,  weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_task)


    def train(self):



        ac = [0.]
        best_model_dict = {}



        # local training (E1)
        for i in range(self.args.epochs):

            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch) ##7s
        # #
            # update learning rate


            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])

        #
            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()


        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return  loc_w,best_model_dict['C']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()


        randomn = torch.rand(y.size(0), self.args.input_size).cuda()
        y_onehot = torch.zeros(y.size(0), self.args.class_num).cuda()
        y_onehot.scatter_(1, y.view(-1, 1), 0.7).cuda()


        self.fetExtrac.train()
        self.classifier.train()

        self.opti_task.zero_grad()


        fake1 = self.generator1(y_onehot, randomn)
        fake2 = self.generator2(y_onehot, randomn)

        feature = self.fetExtrac(x)

        pre = self.classifier(feature)

        pre1 = self.classifier(fake1)
        pre2 = self.classifier(fake2)



        loss_cla = self.lossFunc(pre, y)+self.lossFunc(pre1, y)+self.lossFunc(pre2, y)


        loss_cla.backward()
        self.opti_task.step()



        return loss_cla.item()