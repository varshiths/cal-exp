#!/usr/bin/python3

import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np 
import seaborn as sns

import argparse


from insights_utils import transform_line 
from insights_utils import get_intervals, get_acc_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True,
                    help='The path to the output file for main model')
parser.add_argument('--hinged', type=bool, default=True,
                    help='Whether model outputs extra logit')
parser.add_argument('--T', type=float, default=1.0,
                    help='Temperature for T Scaling')
parser.add_argument('--N', type=int, default=20,
                    help='Number of buckets to measure calibration error')
args = parser.parse_args()

EPSILON=1e-7

N = args.N
NCLASSES = 10
_NCLASSES = NCLASSES + int(args.hinged)

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

    probs = probs[:, :NCLASSES]

    preds = np.argmax(probs, axis=1)
    mprobs = np.max(probs, axis=1)
    mlogits = np.max(logits, axis=1)
    # after T Scaling
    mprobsT = np.max(probs/T, axis=1)

    accur, freq, confs = get_accuracies_and_frequencies(mprobs, preds, targets, N)
    accurT, freqT, _ = get_accuracies_and_frequencies(mprobsT, preds, targets, N)

    ece  = get_ece(mprobs, preds, targets, N)
    eceT = get_ece(mprobsT, preds, targets, N)

    print("ECE: BIN   : %f" % (ece*100))
    print("ECE: BIN+T : %f" % (eceT*100))

    # plot_calibration_graphs(accur, freq, accurT, freqT, confs)
    pass

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

if __name__ == '__main__':
    main()