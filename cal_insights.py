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
    # plt.ylim(-0.02, 1.02)
    plt.plot(confs, freq / np.sum(freq), label="BIN")
    plt.plot(confs, freqT / np.sum(freqT), label="BIN+T")
    # plt.plot(confs, freq, label="BIN")
    # plt.plot(confs, freqT, label="BIN+T")
    plt.legend(loc="upper left")
    plt.savefig("calf.png")

    print("Frequency and Bucket Accuracy in files: cala.png, calf.png")
