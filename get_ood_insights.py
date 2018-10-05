#!/usr/bin/python3

import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np 
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True,
                    help='The path to the output file for main model')
parser.add_argument('--hinged', type=bool, default=True,
                    help='Whether model outputs extra logit')
args = parser.parse_args()

EPSILON=1e-7

if args.hinged:
    _NCLASSES = 11
else:
    _NCLASSES = 10
_OOD_CLASS = 10

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

tensors_to_get = ["Targets[", "Probs[", "Logits["]

def get_tensors_from_file(filename):

    tsrsd = {}
    for tsr in tensors_to_get:
        tsrsd[tsr] = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            for tsr in tensors_to_get:
                if tsr in line:
                    tsrsd[tsr].append(transform_line(line, tsr))
            line = f.readline()

    tsrsd = { x: np.concatenate(y, axis=0) for x, y in tsrsd.items() }

    targets = tsrsd[tensors_to_get[0]]
    probs = tsrsd[tensors_to_get[1]]
    logits = tsrsd[tensors_to_get[2]]

    # perform the required reshapes
    targets = targets
    probs = np.reshape(probs, [-1, _NCLASSES])
    logits = np.reshape(logits, [-1, _NCLASSES])

    assert targets.shape[0] == probs.shape[0], "corrupt: %s" % filename
    return targets, probs, logits

def main():

    targets, probs, logits = get_tensors_from_file(args.file)

    preds = np.argmax(probs, axis=1)
    mprobs = np.max(probs, axis=1)
    mlogits = np.max(logits, axis=1)

    omask = (targets == _OOD_CLASS).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    accuracy = np.sum(cmask * (1-omask)) / np.sum(1-omask)
    detection = np.sum(cmask * omask) / np.sum(omask)

    pcmask = (preds == _OOD_CLASS).astype(np.float32)
    misdetection = np.sum(pcmask * (1-cmask)) / np.sum(pcmask)

    print("Accuracy: ", 100*accuracy if np.sum(1-omask) != 0 else "No samples")

    distr_logits(logits, 1-omask, "IND")
    distr_logits(logits, omask, "OOD")

    # FPR at 95% TPR
    FPR, TPR, threshold = FPR_for_TPR(targets, probs, 0.95, tolerance=1e-3)
    print("At TPR: %f, FPR: %f" % (100*TPR, 100*FPR))
    print("Detection Error: %f" % ( 100*(1-TPR+FPR)/2 ))

    roc = ROC(probs, targets)
    print("AUROC: %f" % ( 100*area_under(roc) ))
    prin = PR(probs, targets)
    print("AUPR-IN: %f" % ( 100*area_under(prin) ))
    prout = PR(probs, targets, True)
    print("AUPR-OUT: %f" % ( 100*(1-area_under(prout)) ))

    # sns.lineplot(prout[:, 0], prout[:, 1]).get_figure().savefig("aupr_out.png")

def area_under(_points):
    points = _points.copy()
    points.sort(axis=0)
    areas = (points[1:, 0]-points[:-1, 0])*(points[:-1, 1]+points[1:, 1])/2
    return np.sum(areas)

def ROC(probs, targets, step_size=1e-3):
    # modify targets and probs for just in/out detection
    targets = (targets != _OOD_CLASS).astype(np.float32)
    probs = 1-probs[:, _OOD_CLASS]

    npts = int(1/step_size)
    points = np.zeros((npts, 2))

    for i in range(npts):
        points[i, 0] = FPR(probs, targets, i*step_size)
        points[i, 1] = TPR(probs, targets, i*step_size)

    return points

def PR(probs, targets, reverse=False, step_size=1e-3):
    # modify targets and probs for just in/out detection
    targets = (targets != _OOD_CLASS).astype(np.float32)
    probs = 1-probs[:, _OOD_CLASS]

    if reverse:
        targets = 1-targets
        probs = 1-probs

    npts = int(1/step_size)
    points = np.zeros((npts, 2))

    for i in range(npts):
        points[i, 0] = Recall(probs, targets, i*step_size)
        points[i, 1] = Precision(probs, targets, i*step_size)

    return points

def Recall(*args):
    return TPR(*args)

def Precision(probs, targets, thresh):
    # assumes targets == 1 is the positive class
    preds = (probs >= thresh).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    TP = np.sum((preds == 1).astype(np.float32)*cmask)
    FP = np.sum((preds == 1).astype(np.float32)*(1-cmask))

    if TP == 0 and FP == 0:
        # print("No points with probs above", thresh)        
        FP = 1.0
    return TP / (TP + FP)

def TPR(probs, targets, thresh):
    # assumes targets == 1 is the positive class
    preds = (probs >= thresh).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    TP = np.sum((preds == 1).astype(np.float32)*cmask)
    FN = np.sum((preds == 0).astype(np.float32)*(1-cmask))
    return TP / (FN + TP)

def FPR(probs, targets, thresh):
    # assumes targets == 1 is the positive class
    preds = (probs >= thresh).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    FP = np.sum((preds == 1).astype(np.float32)*(1-cmask))
    TN = np.sum((preds == 0).astype(np.float32)*cmask)
    return FP / (FP + TN)

def FPR_for_TPR(targets, probs, rate, tolerance=1e-3):
    # modify targets and probs for just in/out detection
    targets = (targets != _OOD_CLASS).astype(np.float32)
    probs = 1-probs[:, _OOD_CLASS]

    # function that guides thresh 1-TPR
    def binary_thresh_search(l, r, func, val):
        mid = l+(r-l)/2
        if abs(func(mid) - val) < tolerance:
            return mid
        elif func(mid) > val:
            return binary_thresh_search(l, mid, func, val)
        else:
            return binary_thresh_search(mid, r, func, val)

    threshold = binary_thresh_search(l=0, r=1, func=lambda x: 1-TPR(probs, targets, x), val=1-rate)
    return FPR(probs, targets, threshold), TPR(probs, targets, threshold), threshold

def distr_logits(logits, mask, name):

    slgts = logits[np.where(mask)]
    if np.sum(mask) != 0:
        _min = np.mean(np.min(slgts, axis=1), axis=0)
        _max = np.mean(np.max(slgts, axis=1), axis=0)
        _mean = np.mean(np.mean(slgts, axis=1), axis=0)
        _std = np.mean(np.std(slgts, axis=1), axis=0)
        print("%s: Min, Max, Mean, Std : %.2f, %.2f, %.2f, %.2f " % (name, _min, _max, _mean, _std))
    else:
        print("%s: No Samples" % (name))

if __name__ == '__main__':
    main()
