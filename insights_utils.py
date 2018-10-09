
import numpy as np

EPSILON = 1e-10
INF = 1e10

def get_intervals(N):
    int0 = np.arange(N)
    int1 = int0+1
    ints = np.stack([int0, int1], axis=1)/N
    return ints, int0

def get_acc_bucket(probs, preds, tgts, confint):
    # mask = (confint[0] <= probs and probs < confint[1]).astype(int)
    mask = np.logical_and(confint[0] <= probs, probs < confint[1]).astype(int)
    if np.sum(mask) == 0:
        accuracy = None
    else:
        accuracy = np.sum((tgts == preds).astype(float)*mask) / np.sum(mask)
    return accuracy, np.sum(mask)

def transform_line(tline, skey):
    tline = tline[tline.index(skey)+len(skey):]
    tline = tline.replace('][', ' ')
    tline = tline.replace(']', '')
    tline = tline.replace('[', '')
    tline = tline.replace('\n', '')
    tline = tline.split(' ')

    tline = [float(x) for x in tline]
    tline = np.array(tline)

    return tline

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
