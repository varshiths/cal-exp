#!/usr/bin/python3

import numpy as np 
import argparse

from insights_utils import get_tensors_from_file
from ood_insights import distr_logits, FPR_for_TPR, ROC, PR, area_under
from cal_insights import get_ece, get_accuracies_and_frequencies, plot_calibration_graphs

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

if args.hinged:
    _NCLASSES = 11
else:
    _NCLASSES = 10
_OOD_CLASS = 10


def main():

    tensors_to_get = ["Targets[", "Probs[", "Logits["]
    targets, probs, logits = get_tensors_from_file(args.file, tensors_to_get, _NCLASSES)

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

    print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (
        100*FPR, 
        100*(1-TPR+FPR)/2, 
        100*area_under(roc),
        100*area_under(prin),
        100*(1-area_under(prout)),
        ))

    targets, probs, logits = get_tensors_from_file(args.file)
    probs = probs[:, :NCLASSES]

    # resrict to in class
    inmask = targets != 10
    targets = targets[ inmask ]
    probs = probs[ inmask ]
    logits = logits[ inmask ]

    preds = np.argmax(probs, axis=1)
    mprobs = np.max(probs, axis=1)
    mlogits = np.max(logits, axis=1)
    # after T Scaling
    mprobsT = np.max(probs/args.T, axis=1)

    accur, freq, confs = get_accuracies_and_frequencies(mprobs, preds, targets, N)
    accurT, freqT, _ = get_accuracies_and_frequencies(mprobsT, preds, targets, N)

    ece  = get_ece(mprobs, preds, targets, N)
    eceT = get_ece(mprobsT, preds, targets, N)

    print("ECE: BIN   : %f" % (ece*100))
    print("ECE: BIN+T : %f" % (eceT*100))

    plot_calibration_graphs(accur, freq, accurT, freqT, confs)

if __name__ == '__main__':
    main()
