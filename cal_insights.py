#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt

import numpy as np 
import seaborn as sns

from insights_utils import get_intervals, get_acc_bucket

def get_ece(confs, preds, targets, N):

    et = 0
    for cint in get_intervals(N)[0]:
        mask = np.logical_and(cint[0] <= confs, confs < cint[1]).astype(int)
        et += np.absolute(np.sum(((targets == preds).astype(float) - confs)*mask))

    return et / confs.shape[0]

def get_accuracies_and_frequencies(mprobs, preds, targets, N):

    ints, indices = get_intervals(N)
    accur = np.zeros(N)
    freq = np.zeros(N)

    for i, cint in enumerate(ints):
        tp = get_acc_bucket(mprobs, preds, targets, cint)
        accur[i] = tp[0]
        freq[i] = tp[1]

    return accur, freq, indices/N

def plot_calibration_graphs(accur, freq, accurT, freqT, confs):

    # plot accuracies in buckets of confidences
    plt.figure()
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot(confs, confs, label="Reference")
    plt.plot(confs, accur, label="BIN")
    plt.plot(confs, accurT, label="BIN+T")
    plt.legend(loc="upper left")
    plt.savefig("cala.png")

    # plot histogram of confidences of outputs
    fig = plt.figure()
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot(confs, freq / np.sum(freq), label="BIN")
    plt.plot(confs, freqT / np.sum(freqT), label="BIN+T")
    plt.legend(loc="upper left")
    plt.savefig("calf.png")
