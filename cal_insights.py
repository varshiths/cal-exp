#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt

import numpy as np 
import seaborn as sns

from insights_utils import get_intervals, get_acc_bucket

def get_ece(targets, imask, preds, confs, N):

    et = 0
    for cint in get_intervals(N)[0]:
        mask = np.logical_and(cint[0] <= confs, confs < cint[1]).astype(float)*imask
        et += np.absolute(np.sum(((targets == preds).astype(float) - confs)*mask))

    return et / np.sum(imask)

def get_accuracies_and_frequencies(targets, imask, preds, confs, N):

    ints, indices = get_intervals(N)
    accur = np.zeros(N)
    freq = np.zeros(N)

    for i, cint in enumerate(ints):
        mask = np.logical_and(cint[0] <= confs, confs < cint[1]).astype(int)*imask
        accur[i] = np.sum((targets == preds).astype(float)*mask) / np.sum(mask) if np.sum(mask) != 0 else None
        freq[i] = np.sum(mask)

    return accur, freq, indices/N

def get_per_class_ece(_class, targets, preds, confs, N):

    # ex_targets = np.expand_dims(targets, axis=1)    
    # ktars = np.tile( ex_targets, [1, 10, 1] )
    # ktars = np.transpose(ktars, [0, 2, 1])
    # ktars = np.squeeze(ktars, 1)
    imask = np.expand_dims((targets == _class).astype(np.float32), axis=1)

    targets = (np.eye(11)[targets.astype(np.int32)])[:, :10]
    # targets = np.reshape(targets, [-1])
    # confs = np.reshape(confs, [-1])

    et = 0
    for cint in get_intervals(N)[0]:
        mask = np.logical_and(cint[0] <= confs, confs < cint[1]).astype(float) * imask
        et += np.absolute(np.sum(((targets == preds).astype(float) - confs)*mask))

    return et / np.sum(imask)

def get_per_class_accuracies_and_frequencies(_class, targets, preds, confs, N):

    # ex_targets = np.expand_dims(targets, axis=1)
    # ktars = np.tile( ex_targets, [1, 10, 1] )
    # ktars = np.transpose(ktars, [0, 2, 1])
    # ktars = np.squeeze(ktars, 1)
    imask = np.expand_dims((targets == _class).astype(np.float32), axis=1)

    targets = (np.eye(11)[targets.astype(np.int32)])[:, :10]

    # import pdb; pdb.set_trace()

    # targets = np.reshape(targets, [-1])
    # confs = np.reshape(confs, [-1])

    ints, indices = get_intervals(N)
    accur = np.zeros(N)
    freq = np.zeros(N)

    for i, cint in enumerate(ints):
        mask = np.logical_and(cint[0] <= confs, confs < cint[1]).astype(float) * imask
        accur[i] = np.sum((targets == preds).astype(float)*mask) / np.sum(mask) if np.sum(mask) != 0 else None
        freq[i] = np.sum(mask)

    return accur, freq, indices/N

def plot_calibration_graphs(accur, freq, accurT, freqT, confs, sufx):
    sufx = str(sufx)

    # plot accuracies in buckets of confidences
    plt.figure()
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot(confs, confs, label="Reference")
    plt.bar(x=confs, height=accur, width=0.01, label="BIN")
    # plt.bar(x=confs+0.02, height=accurT, width=0.01, label="BIN+T")
    plt.legend(loc="upper left")
    plt.savefig("figures/"+sufx+"_cal_accur.png")

    # plot histogram of confidences of outputs
    fig = plt.figure()
    plt.xlim(-0.02, 1.02)
    # plt.ylim(-0.02, 1.02)
    plt.bar(x=confs, height=freq, width=0.01, label="BIN")
    # plt.bar(x=confs+0.02, height=freqT, width=0.01, label="BIN+T")
    plt.legend(loc="upper left")
    plt.savefig("figures/"+sufx+"_cal_freq.png")

    print("Frequency and Bucket Accuracy in files in figures/\"sufx\" + : _cal_accur.png, _cal_freq.png")
