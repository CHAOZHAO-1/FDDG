import copy
import torch

# server perform aggregation parameters
def FedAvg(w):
    """
    :param w: paramrters (w_f, w_c, w_g) uploaded by clients.
    :return: new parameters after aggregation.
    """
    l_model = len(w[0])  # number of model
    l_user = len(w)      # the number of user
    w_s, w_avgs = [], []
    for i in range(l_model):
        w_s.append([])
        for k in range(l_user):
            w_s[i].append(w[k][i])
        w_avgs.append(CountAvg(w_s[i]))
    return w_avgs

def CountAvg(w):
    w_avg = copy.deepcopy(w[0])
    l = len(w)
    for k in w_avg.keys():
        for i in range(1, l):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], l)
    return w_avg