
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('GTK')
plt.style.use('ggplot')

import numpy as np 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_main', type=str, required=True,
                    help='The path to the output file for main model')
parser.add_argument('--file_base', type=str, required=True,
                    help='The path to the output file for baseline model')
parser.add_argument('--file_out', type=str, required=True,
                    help='The name of the png file of hist')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='The temperature to perform T scaling')
parser.add_argument('--N', type=int, default=20,
                    help='Number of bins')
args = parser.parse_args()

EPSILON=1e-7

N = args.N
_NCLASSES = 11

def get_bin_num(val):
    global N
    val = min(int(val*N), N)
    return val

def softmax(arr):
    arr1 = arr - np.expand_dims(np.max(arr, 1), 1)
    return np.exp(arr1)/np.expand_dims(np.sum(np.exp(arr1), 1), 1)

def transform_target_line(tline):
    tline = tline[tline.index('Targets[')+8:]
    tline = tline.replace('][', ' ')
    tline = tline.replace(']', '')
    tline = tline.replace('[', '')
    tline = tline.replace('\n', '')
    tline = tline.split(' ')

    tline = [int(x) for x in tline]
    tline = np.array(tline)

    return tline

def transform_probs_line(pline):
    pline = pline[pline.index('Probs[')+6:]
    pline = pline.replace('][', ' ')
    pline = pline.replace(']', '')
    pline = pline.replace('[', '')
    pline = pline.replace('\n', '')
    pline = pline.split(' ')
    pline = [float(x) for x in pline]
    
    pline = np.array(pline)
    pline = np.reshape(pline, [-1, _NCLASSES])

    return pline

def transform_logits_line(lline):
    lline = lline[lline.index('Logits[')+7:]
    lline = lline.replace('][', ' ')
    lline = lline.replace(']', '')
    lline = lline.replace('[', '')
    lline = lline.replace('\n', '')
    lline = lline.split(' ')
    lline = [float(x) for x in lline]
    
    lline = np.array(lline)
    lline = np.reshape(lline, [-1, _NCLASSES])

    return lline

def transform_features_line(lline):
    lline = lline[lline.index('Features[')+9:]
    lline = lline.replace('][', ' ')
    lline = lline.replace(']', '')
    lline = lline.replace('[', '')
    lline = lline.replace('\n', '')
    lline = lline.split(' ')
    lline = [float(x) for x in lline]
    
    lline = np.array(lline)
    lline = np.reshape(lline, [-1, 32, 32])

    return lline


def get_targets_and_probs_from_file(filename):
    targets = []
    probs = []
    logits = []
    features = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if 'Targets' in line:
                targets.append(transform_target_line(line))
            if 'Probs' in line:
                probs.append(transform_probs_line(line))
            if 'Logits' in line:
                logits.append(transform_logits_line(line))
            # if 'Features' in line:
            #     features.append(transform_features_line(line))
            line = f.readline()

    targets = np.concatenate(targets, axis=0)
    probs = np.concatenate(probs, axis=0)
    logits = np.concatenate(logits, axis=0)
    # features = np.concatenate(features, axis=0)

    assert targets.shape[0] == probs.shape[0], "corrupt: %s" % filename
    # return targets, probs, logits, features
    return targets, probs, logits, None

def main():

    mmce_list = []
    correct_list = []
    baseline_list = []
    #baseline_temp_list = []
    mmce_prob_values = []
    baseline_prob_values = []
    baseline_temp_prob_values = []

    targets, probs_base, logits_base, features = get_targets_and_probs_from_file(args.file_base)
    _targets, probs_main, logits_main, _ = get_targets_and_probs_from_file(args.file_main)

    assert (targets == _targets).all(), "base and main are not run on the same set"

    pred_base = np.argmax(probs_base, axis=1)
    mprobs_base = np.max(probs_base, axis=1)
    # temperature scaling for base line to manage calibration
    # to correct
    mprobs_base_t = np.max(probs_base, axis=1)

    pred_main = np.argmax(probs_main, axis=1)
    mprobs_main = np.max(probs_main, axis=1)

    mlogits_base = np.max(logits_base, axis=1)
    mlogits_main = np.max(logits_main, axis=1)

    # with open("log_file.txt", "w") as f:
    #     for i in range(10000):
    #         f.write(str(correct_list[i]) + ' ' + str(mmce_list[i]) + ' ' + str(baseline_list[i]) + ' ' + str(mmce_prob_values[i]) + ' ' + str(baseline_prob_values[i]) + ' ' + str(baseline_temp_prob_values[i]) + '\n')

    # ch_base = np.sum(mprobs_base >= 0.99)
    # ch_base_t = np.sum(mprobs_base_t >= 0.99)
    # ch_main = np.sum(mprobs_main >= 0.99)
    # print ('Baseline:       ', ch_base)
    # print ('Baseline + T:   ', ch_base_t)
    # print ('Main:           ', ch_main)

    def get_acc_bucket(probs, preds, tgts, confint):
        # mask = (confint[0] <= probs and probs < confint[1]).astype(int)
        mask = np.logical_and(confint[0] <= probs, probs < confint[1]).astype(int)
        if np.sum(mask) == 0:
            accuracy = None
        else:
            accuracy = np.sum((tgts == preds).astype(float)*mask) / np.sum(mask)
        return accuracy, np.sum(mask)

    def get_acc(prob_list, tgt_list, pred_list):
        avg_accuracy = (np.array(prob_list) >= 0.99)*(np.array(tgt_list) == np.array(pred_list))
        return np.sum(avg_accuracy)

    def get_intervals(N):
        int0 = np.arange(N)
        int1 = int0+1
        ints = np.stack([int0, int1], axis=1)/N
        return ints, int0

    ints, indices = get_intervals(args.N)
    indices = indices/N
    accur = np.zeros(args.N)
    freq_base = np.zeros(args.N)
    freq_main = np.zeros(args.N)
    plt.figure()
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    # baseline, indices=indices
    plt.plot(indices, indices, label="reference")
    
    for i in range(ints.shape[0]):
        accur[i], freq_base[i] = get_acc_bucket(mprobs_base, pred_base, targets, ints[i])
    plt.plot(indices, accur, label="baseline")

    for i in range(ints.shape[0]):
        accur[i], freq_main[i] = get_acc_bucket(mprobs_main, pred_main, targets, ints[i])
    plt.plot(indices, accur, label="main")

    plt.legend(loc="upper left")
    plt.savefig(args.file_out)
    # plt.show()

    # plot histogram of confidences of outputs
    fig = plt.figure()

    # fig.add_subplot(2, 1, 1)
    plt.plot(indices, freq_base / np.sum(freq_base), label="baseline")
    # fig.add_subplot(2, 1, 2)
    plt.plot(indices, freq_main / np.sum(freq_main), label="main")
    plt.legend(loc="upper left")

    plt.savefig("hist_" + args.file_out)

if __name__ == '__main__':
    main()