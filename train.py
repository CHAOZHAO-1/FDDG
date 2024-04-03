import argparse
import copy
import random
import numpy
import torch
import os
from torch import optim
from localTrain import localTrain_1,localTrain_2,localTrain_3,localTrain_4
from Fed import FedAvg
from Nets import task_classifier, GeneDistrNet, Discriminator, feature_extractor
from test import test1_g,test1_gg
import numpy as np
import time
from sampling import  get_dataset
from utilss import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def args_parser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--fft1', type=str, default=False, help="time samples or frequency samples")
    paser.add_argument('--class_num', type=int, default=3, help="number of classes")
    paser.add_argument('--dataset', type=str, default='SQgearbox', help='name of dataset')
    paser.add_argument('--batch-size', type=int, default=256, help='batch size for training')
    paser.add_argument('--workers', type=int, default=4, help='number of data-loading workers')
    paser.add_argument('--pin', type=bool, default=True, help='pin-memory')
    paser.add_argument('--lr0', type=float, default=0.005, help='learning rate 0')## feature extractor and classifier's learning rate
    paser.add_argument('--lr1', type=float, default=0.3, help='learning rate 1')### generator leanring rate
    paser.add_argument('--lr2', type=float, default=0.01, help='learning rate 2')
    paser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    paser.add_argument('--weight-dec', type=float, default=1e-5, help='0.005weight decay coefficient default 1e-5')
    paser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size 1024')
    paser.add_argument('--epochs', type=int, default=1, help='rounds of training')
    paser.add_argument('--current_epoch', type=int, default=1, help='current epoch in training')
    paser.add_argument('--factor', type=float, default=0.2, help='lr decreased factor (0.1)')
    paser.add_argument('--patience', type=int, default=20, help='number of epochs to want before reduce lr (20)')
    paser.add_argument('--lr-threshold', type=float, default=1e-4, help='lr schedular threshold')
    paser.add_argument('--ite-warmup', type=int, default=100, help='LR warm-up iterations (default:500)')
    paser.add_argument('--label_smoothing', type=float, default=0.1, help='the rate of wrong label(default:0.2)')
    paser.add_argument('--input_size', type=int, default=128, help='the size of hidden feature')
    paser.add_argument('--hidden_size', type=int, default=256, help='the size of hidden feature')
    paser.add_argument('--global_epochs', type=int, default=200, help='the num of global train epochs')
    paser.add_argument('--i_epochs', type=int, default=2, help='the num of independent epochs in local')
    paser.add_argument('--path_root', type=str, default='../data/PACS/', help='the root of dataset')
    args = paser.parse_args()
    return args
if __name__ == '__main__':
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    numpy.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    src_tar = np.array([[0, 6, 12, 23], [1, 7, 13, 23], [2, 8, 14, 23], [0, 6, 12, 29], [1, 7, 13, 29],
                        [6, 12, 18, 29], [0, 12, 18, 29], [12, 18, 24, 11], [0, 18, 24, 11], [0, 12, 18, 11]
                        ])

    a = 40

    b = 60


    for taskindex in range(10):
        source1 = src_tar[taskindex][0]
        source2 = src_tar[taskindex][1]
        source3 = src_tar[taskindex][2]
        target = src_tar[taskindex][3]

        for repeat in range(10):

            Train_Loss_list = []
            Train_Accuracy_list = []
            Test_Loss_list = []
            Test_Accuracy_list = []


            start = time.time()

            src_name1 = 'load' + str(source1) + '_train'
            src_name2 = 'load' + str(source2) + '_train'
            src_name3 = 'load' + str(source3) + '_train'
            test_name = 'load' + str(target) + '_test'

            client = [src_name1, src_name2, src_name3, test_name]

            torch.cuda.empty_cache()

            train_loaders, target_loader = get_dataset(args, client)  ### Valid_loaders？？

            # initial the global model
            global_fetExtrac = feature_extractor(optim.Adam, args.lr0, args.weight_dec)

            Domain_fetExtrac = feature_extractor(optim.Adam, args.lr0, args.weight_dec)

            # global_fetExtrac.load_state_dict(load_FCparas("alexnet"), strict=False)

            global_fetExtrac.optimizer = optim.Adam(global_fetExtrac.parameters(), args.lr0,weight_decay=args.weight_dec)

            global_classifier_1 = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.weight_dec,
                                                class_num=args.class_num)
            global_classifier_2 = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.weight_dec,
                                                class_num=args.class_num)
            global_classifier_3 = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.weight_dec,
                                                class_num=args.class_num)

            Domain_classifier= task_classifier(args.hidden_size, optim.SGD, args.lr0, args.weight_dec,
                                                  class_num=3)



            global_generator_1 = GeneDistrNet(args.input_size, args.hidden_size, optim.Adam, args.lr1,args.weight_dec)
            global_generator_2 = GeneDistrNet(args.input_size, args.hidden_size, optim.Adam, args.lr1, args.weight_dec)
            global_generator_3 = GeneDistrNet(args.input_size, args.hidden_size, optim.Adam, args.lr1, args.weight_dec)

            local_cc = []

            for i in range(3):  # local discriminator

                global_c = task_classifier(args.hidden_size, optim.Adam, args.lr0, args.weight_dec,
                                           class_num=args.class_num)
                global_c.optimizer = optim.Adam(global_c.parameters(), args.lr1, weight_decay=args.weight_dec)

                local_cc.append(global_c)





            # server execution phase

            models_global = []

            model_best_paras, best_acc, best_id = {}, 0., 0
            for t in range(args.global_epochs):

                ##########Stage 1########################################
                if t<=a:

                    print('global training epoch: %d ' % (t))
                    args.current_epoch = t + 1
                    w_locals = []
                    # client update

                    list_acc = []
                    for i in range(3):
                        if i==0:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_c = copy.deepcopy(global_classifier_1)

                            trainer = localTrain_1(local_f, local_c, train_loaders[i], args)

                            w,wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                            w_locals.append(w)
                            local_cc[i]=wc

                        if i == 1:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_c = copy.deepcopy(global_classifier_2)

                            trainer = localTrain_1(local_f, local_c, train_loaders[i], args)

                            w, wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                            w_locals.append(w)
                            local_cc[i] = wc

                        if i == 2:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_c = copy.deepcopy(global_classifier_3)

                            trainer = localTrain_1(local_f, local_c, train_loaders[i], args)

                            w, wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                            w_locals.append(w)
                            local_cc[i] = wc

                    models_global.clear()
                    # aggregation
                    models_global = FedAvg(w_locals)

                    global_fetExtrac.load_state_dict(models_global[0])

                    Domain_fetExtrac.load_state_dict(models_global[0])

                    global_classifier_1.load_state_dict(local_cc[0])
                    global_classifier_2.load_state_dict(local_cc[1])
                    global_classifier_3.load_state_dict(local_cc[2])

                    acc_target = test1_g(global_fetExtrac, global_classifier_1,global_classifier_2,global_classifier_3, target_loader)



                ##########Stage 2############################################
                if a<t<=b:
                    print('global training epoch: %d ' % (t))
                    args.current_epoch = t + 1
                    w_locals = []
                    # client update

                    list_acc = []
                    for i in range(3):
                        if i == 0:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_g = copy.deepcopy(global_generator_1)
                            local_c = copy.deepcopy(global_classifier_1)

                            trainer = localTrain_2(local_f, local_g,local_c, train_loaders[i], args)

                            wg1 = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier


                        if i == 1:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_g = copy.deepcopy(global_generator_2)
                            local_c = copy.deepcopy(global_classifier_2)

                            trainer = localTrain_2(local_f, local_g, local_c, train_loaders[i], args)

                            wg2 = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                        if i == 2:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_g = copy.deepcopy(global_generator_3)
                            local_c = copy.deepcopy(global_classifier_3)

                            trainer = localTrain_2(local_f, local_g, local_c, train_loaders[i], args)


                            wg3 = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier



                    global_generator_1.load_state_dict(wg1)
                    global_generator_2.load_state_dict(wg2)
                    global_generator_3.load_state_dict(wg3)


                ###########Stage 3#################################################################

                if t > b:
                    print('global training epoch: %d ' % (t))
                    args.current_epoch = t + 1
                    w_locals = []
                    # client update

                    list_acc = []
                    for i in range(3):
                        if i == 0:
                            print('source domain {}/3'.format(i + 1))

                            local_f = copy.deepcopy(global_fetExtrac)

                            local_g2 = copy.deepcopy(global_generator_2)
                            local_g3 = copy.deepcopy(global_generator_3)

                            local_c = copy.deepcopy(global_classifier_1)


                            trainer = localTrain_4(local_f, local_g2,local_g3, local_c, train_loaders[i], args)

                            w, wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                            w_locals.append(w)
                            local_cc[i] = wc


                        if i == 1:
                            print('source domain {}/3'.format(i + 1))

                            local_f = copy.deepcopy(global_fetExtrac)

                            local_g1 = copy.deepcopy(global_generator_1)
                            local_g3 = copy.deepcopy(global_generator_3)

                            local_c = copy.deepcopy(global_classifier_2)

                            trainer = localTrain_4(local_f, local_g1, local_g3, local_c, train_loaders[i], args)

                            w, wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier

                            w_locals.append(w)
                            local_cc[i] = wc

                        if i == 2:
                            print('source domain {}/3'.format(i + 1))

                            local_f = copy.deepcopy(global_fetExtrac)

                            local_g1 = copy.deepcopy(global_generator_1)
                            local_g2 = copy.deepcopy(global_generator_2)

                            local_c = copy.deepcopy(global_classifier_3)

                            trainer = localTrain_4(local_f, local_g1, local_g2, local_c, train_loaders[i], args)


                            w, wc = trainer.train()  ###这一步训练很慢##返回 feature extractor and classifier
                            w_locals.append(w)
                            local_cc[i] = wc


                    models_global.clear()
                    # aggregation
                    models_global = FedAvg(w_locals)

                    global_fetExtrac.load_state_dict(models_global[0])

                    Domain_fetExtrac.load_state_dict(models_global[0])

                    global_classifier_1.load_state_dict(local_cc[0])
                    global_classifier_2.load_state_dict(local_cc[1])
                    global_classifier_3.load_state_dict(local_cc[2])

                    acc_target = test1_g(global_fetExtrac, global_classifier_1, global_classifier_2,
                                         global_classifier_3, target_loader)



















    #



