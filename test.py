import os
import copy
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Test on target domain data
def test1_g(fetExtrac, classifier_1, classifier_2, classifier_3, valid_loader):
    fetExtrac = fetExtrac.eval()

    classifier_1 = classifier_1.eval()
    classifier_2 = classifier_2.eval()
    classifier_3 = classifier_3.eval()



    num_correct_all = 0.
    num_correct_mean = 0.
    num_correct_1 = 0.
    num_correct_2 = 0.
    num_correct_3 = 0.


    num_all = 0.

    m = nn.Softmax(dim=1)
    with torch.no_grad():
        fetExtrac = fetExtrac.cuda()
        classifier_1 = classifier_1.cuda()
        classifier_2 = classifier_2.cuda()
        classifier_3 = classifier_3.cuda()




        for t, batch in enumerate(valid_loader, 0):
            x, y = batch
            x = x.cuda()
            y = y.cuda()

            feature = fetExtrac(x)

            label_out_1 = classifier_1(feature)
            label_out_2 = classifier_2(feature)
            label_out_3 = classifier_3(feature)

            num_all += x.size(0)



            pred_mean = m(label_out_1) + m(label_out_2) + m(label_out_3)
            pred_mean = pred_mean.data.max(1)[1]
            num_correct_mean += pred_mean.eq(y.data.view_as(pred_mean)).cpu().sum().numpy()



            pre_1 = torch.max(label_out_1, 1)[1].data.squeeze()

            num_correct_1 += pre_1.eq(y.data.view_as(pre_1)).cpu().sum().numpy()

            pre_2 = torch.max(label_out_2, 1)[1].data.squeeze()

            num_correct_2 += pre_2.eq(y.data.view_as(pre_2)).cpu().sum().numpy()

            pre_3 = torch.max(label_out_3, 1)[1].data.squeeze()

            num_correct_3 += pre_3.eq(y.data.view_as(pre_3)).cpu().sum().numpy()





        acc_mean = num_correct_mean / num_all
        acc_1 = num_correct_1 / num_all
        acc_2 = num_correct_2 / num_all
        acc_3 = num_correct_3 / num_all

        print('test mean :{} acc1:{}  acc2:{} acc3:{}'.format(acc_mean, acc_1,acc_2,acc_3))

    return acc_mean


def test1_gg(fetExtrac, classifier_1, classifier_2, classifier_3,domain_feaExtrac,domain_classifier, valid_loader):
    fetExtrac = fetExtrac.eval()
    domain_feaExtrac = domain_feaExtrac.eval()

    classifier_1 = classifier_1.eval()
    classifier_2 = classifier_2.eval()
    classifier_3 = classifier_3.eval()
    domain_classifier = domain_classifier.eval()



    num_correct_vote = 0.
    num_correct_mean = 0.
    num_correct_1 = 0.
    num_correct_2 = 0.
    num_correct_3 = 0.


    num_all = 0.

    m = nn.Softmax(dim=1)
    with torch.no_grad():
        fetExtrac = fetExtrac.cuda()
        domain_feaExtrac = domain_feaExtrac.cuda()

        classifier_1 = classifier_1.cuda()
        classifier_2 = classifier_2.cuda()
        classifier_3 = classifier_3.cuda()
        domain_classifier = domain_classifier.cuda()




        for t, batch in enumerate(valid_loader, 0):
            x, y = batch
            x = x.cuda()
            y = y.cuda()

            feature = fetExtrac(x)

            domain_fea=domain_feaExtrac(x)

            label_out_1 = classifier_1(feature)
            label_out_2 = classifier_2(feature)
            label_out_3 = classifier_3(feature)

            domain_pre=domain_classifier(domain_fea)

            domain_pre=m(domain_pre)

            print(torch.sum(domain_pre,dim=0)/y.size(0))



            num_all += x.size(0)

            pred_vote = m(label_out_1) * domain_pre[:, 0].reshape(-1, 1) \
                        + m(label_out_2) * domain_pre[:, 1].reshape(-1, 1) \
                        + m(label_out_3) * domain_pre[:, 2].reshape(-1, 1)

            pred_vote = pred_vote.data.max(1)[1]
            num_correct_vote += pred_vote.eq(y.data.view_as(pred_vote)).cpu().sum().numpy()



            pred_mean = m(label_out_1) + m(label_out_2) + m(label_out_3)
            pred_mean = pred_mean.data.max(1)[1]
            num_correct_mean += pred_mean.eq(y.data.view_as(pred_mean)).cpu().sum().numpy()



            pre_1 = torch.max(label_out_1, 1)[1].data.squeeze()

            num_correct_1 += pre_1.eq(y.data.view_as(pre_1)).cpu().sum().numpy()

            pre_2 = torch.max(label_out_2, 1)[1].data.squeeze()

            num_correct_2 += pre_2.eq(y.data.view_as(pre_2)).cpu().sum().numpy()

            pre_3 = torch.max(label_out_3, 1)[1].data.squeeze()

            num_correct_3 += pre_3.eq(y.data.view_as(pre_3)).cpu().sum().numpy()


        acc_vote=num_correct_vote / num_all
        acc_mean = num_correct_mean / num_all
        acc_1 = num_correct_1 / num_all
        acc_2 = num_correct_2 / num_all
        acc_3 = num_correct_3 / num_all

        print('test vote :{} test mean :{} acc1:{}  acc2:{} acc3:{}'.format(acc_vote,acc_mean, acc_1,acc_2,acc_3))

    return acc_mean