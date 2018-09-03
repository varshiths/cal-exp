#!/usr/bin/python3

import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np 

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

    print("Accuracy: ", accuracy if np.sum(1-omask) != 0 else "No samples")
    print("Detection: ", detection if np.sum(omask) != 0 else "No samples")
    print("Misdetection: ", misdetection if np.sum(pcmask) != 0 else "No samples")

    distr_logits(logits, 1-omask, "IND")
    distr_logits(logits, omask, "OOD")

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
