#!/usr/bin/python3

import numpy as np 
import argparse

from insights_utils import get_tensors_from_file, softmax, sigmoid
from ood_insights import distr_logits, FPR_for_TPR, ROC, PR, area_under, distr_rates
from cal_insights import get_ece, get_per_class_ece, get_accuracies_and_frequencies, plot_calibration_graphs

from cal_insights import get_per_class_ece, get_per_class_accuracies_and_frequencies

from ood_insights import area_under_ROC, area_under_PR

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True,
                    help='The path to the output file for main model')
parser.add_argument('--hinged', type=bool, default=True,
                    help='Whether model outputs extra logit')
parser.add_argument('--nbin', type=float, default=None,
                    help='Whether probs and rate need to be calc from logits'
                    'None: Consider the given probs and rate'
                    '<float>: Calc probs and rate from logits using this as temp')
parser.add_argument('--nbaset', type=float, default=None,
                    help='Whether probs and rate need to be calc from logits'
                    'None: Consider the given probs and rate'
                    '<float>: Calc probs and rate from logits using this as temp for base scaling')
parser.add_argument('--T', type=float, default=1.0,
                    help='Temperature for T Scaling')
parser.add_argument('--N', type=int, default=20,
                    help='Number of buckets to measure calibration error')
args = parser.parse_args()

EPSILON=1e-7

# if args.hinged:
    # _NCLASSES = 11
# else:
_NCLASSES = 10
_OOD_CLASS = 10


def main():

    # tensors_to_get = ["Targets[", "Rate[", "Confs[", "Probs[", "Logits["]
    # targets, rate, bin_probs, probs, logits = get_tensors_from_file(args.file, tensors_to_get, _NCLASSES, num_non_reshape=2)
    tensors_to_get = ["Targets[", "Rate[", "Probs[", "Logits["]
    targets, rate, probs, logits = get_tensors_from_file(args.file, tensors_to_get, _NCLASSES, num_non_reshape=2)
    bin_probs = probs

    # import pdb; pdb.set_trace()

    if args.nbin is not None:
        probs = sigmoid(logits/args.nbin)
        rate = np.max(probs, axis=-1)

    if args.nbaset is not None:
        probs = softmax(logits/args.nbaset)
        rate = np.max(probs, axis=-1)

    # import pdb; pdb.set_trace()

    # rate = np.max(bin_probs, axis=1)
    preds = np.argmax(logits, axis=1)

    omask = (targets == _OOD_CLASS).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    accuracy = np.sum(cmask * (1-omask)) / np.sum(1-omask)
    print("Accuracy: ", 100*accuracy if np.sum(1-omask) != 0 else "No samples")

    nll = - np.sum( np.log(probs) , axis=-1 )
    print("NLL: ", nll if np.sum(1-omask) != 0 else "No samples")

    # distr_logits(logits, 1-omask, "IND")
    # distr_logits(logits, omask, "OOD")

    # import pdb; pdb.set_trace()

    # modify targets and probs for just in/out detection
    itargets = (targets != _OOD_CLASS).astype(np.float32)
    # FPR at 95% TPR
    FPR, TPR, threshold = FPR_for_TPR(itargets, rate, 0.95, tolerance=1e-3)
    de = (1-TPR+FPR)/2
    print("At TPR: %f, FPR: %f" % (100*TPR, 100*FPR))
    print("Detection Error: %f" % (100*de))

    auroc = area_under_ROC(rate, itargets, step_size=1e-4)
    print("AUROC: %f" % ( 100*auroc ))
    auprin = area_under_PR(rate, itargets, step_size=1e-4)
    print("AUPR-IN: %f" % (100*auprin) )
    auprout = area_under_PR(rate, itargets, reverse=True, step_size=1e-4)
    print("AUPR-OUT: %f" % (100*auprout) )

    # distr_rates(rate, 1-omask, "IND")
    # distr_rates(rate, omask, "OOD")

    print("copy-here->\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (
        100*FPR, 
        100*de,
        100*auroc,
        100*auprin,
        100*auprout,
        ))

    # import pdb; pdb.set_trace()

    # resrict to in class by using imask = 1-omask
    # max of probs is confidence
    # after T Scaling
    # class_confs = np.max(probs, axis=1)
    # class_confsT = np.max(probs/args.T, axis=1)

    # ece  = get_ece(targets, 1-omask, preds, class_confs, args.N)
    # eceT = get_ece(targets, 1-omask, preds, class_confsT, args.N)
    # print("ECE: BIN   : %f" % (ece*100))
    # print("ECE: BIN+T : %f" % (eceT*100))

    # here onwards
    inthresh = 0.5

    io_preds = (rate >= inthresh).astype(np.int32)
    io_confs = rate*io_preds + (1-rate)*(1-io_preds)
    io_targets = (targets != _OOD_CLASS).astype(np.int32)

    ece  = get_ece(io_targets, omask, io_preds, io_confs, args.N)
    print("ECE OODs for OUT: BIN   : %f" % (ece*100))
    # # calibration graph for confidences
    accur, freq, iconfs = get_accuracies_and_frequencies(io_targets, omask, io_preds, io_confs, args.N)
    # accurT, freqT, _ = get_accuracies_and_frequencies(io_targets, omask, io_preds, io_confs, args.N)
    # plot_calibration_graphs(accur, freq, accurT, freqT, iconfs, sufx="out")

    # plot_calibration_graphs(accur, freq, None, None, iconfs, sufx="out")


    # io_preds = (rate >= 0.5).astype(np.int32)
    # io_confs = rate*io_preds + (1-rate)*(1-io_preds)
    # io_targets = (targets != _OOD_CLASS).astype(np.int32)

    ece  = get_ece(io_targets, 1-omask, io_preds, io_confs, args.N)
    print("ECE OODs for IN: BIN   : %f" % (ece*100))
    # # calibration graph for confidences
    accur, freq, iconfs = get_accuracies_and_frequencies(io_targets, 1-omask, io_preds, io_confs, args.N)
    # accurT, freqT, _ = get_accuracies_and_frequencies(io_targets, 1-omask, io_preds, io_confs, args.N)
    # plot_calibration_graphs(accur, freq, accurT, freqT, iconfs, sufx="in")

    # plot_calibration_graphs(accur, freq, None, None, iconfs, sufx="in")
    
    
    if False:
    # if args.nbin is not None:
        # uncomment as required
        amask = np.ones(omask.shape)
        _CLASS = 4
        confc = probs[:, _CLASS]
        io_preds = (confc >= 0.5).astype(np.int32)
        io_confs = confc*io_preds + (1-confc)*(1-io_preds)
        io_targets = (targets == _CLASS).astype(np.int32)

        ece  = get_ece(io_targets, amask, io_preds, io_confs, args.N)
        print("ECE OODs for C4: BIN   : %f" % (ece*100))
        # # calibration graph for confidences
        accur, freq, iconfs = get_accuracies_and_frequencies(io_targets, amask, io_preds, io_confs, args.N)
        # accurT, freqT, _ = get_accuracies_and_frequencies(io_targets, 1-omask, io_preds, io_confs, args.N)
        # plot_calibration_graphs(accur, freq, accurT, freqT, iconfs, sufx=_CLASS)
        plot_calibration_graphs(accur, freq, None, None, iconfs, sufx=_CLASS)
    
if __name__ == '__main__':
    main()
