#!/usr/bin/python3

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
parser.add_argument('--N', type=int, default=20,
                    help='Number of bins')
args = parser.parse_args()

EPSILON=1e-7

N = args.N
_NCLASSES = 11
_OOD_CLASS = 10

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

    targets, probs_main, logits_main, _ = get_targets_and_probs_from_file(args.file_main)

    pred_main = np.argmax(probs_main, axis=1)
    mprobs_main = np.max(probs_main, axis=1)

    mlogits_main = np.max(logits_main, axis=1)

    ood_mask = targets == _OOD_CLASS
    corr_mask = pred_main == targets

    fmask = ood_mask.astype(np.float32)
    dmask = 1 - fmask

    accuracy = np.sum(corr_mask.astype(np.float32) * dmask) / np.sum(dmask)
    detection = np.sum(corr_mask.astype(np.float32) * fmask) / np.sum(fmask)

    print("Accuracy: ", accuracy)
    print("Detection: ", detection)


    def get_acc_bucket(probs, preds, tgts, confint):
        # mask = (confint[0] <= probs and probs < confint[1]).astype(int)
        mask = np.logical_and(confint[0] <= probs, probs < confint[1]).astype(int)
        # mask = np.logical_and(confint[0] <= probs, probs < confint[1], preds != _OOD_CLASS).astype(int)
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
    
    # for i in range(ints.shape[0]):
    #     accur[i], freq_base[i] = get_acc_bucket(mprobs_base, pred_base, targets, ints[i])
    # plt.plot(indices, freq, label="baseline")

    for i in range(ints.shape[0]):
        accur[i], freq_main[i] = get_acc_bucket(mprobs_main, pred_main, targets, ints[i])
    plt.plot(indices, freq_main / np.sum(freq_main), label="frequency")

    plt.plot(indices, accur, label="accuracy")

    plt.legend(loc="upper left")
    plt.savefig("ood_hist.png")
    # plt.show()

    # # plot histogram of confidences of outputs
    # fig = plt.figure()

    # # fig.add_subplot(2, 1, 1)
    # plt.plot(indices, freq_base / np.sum(freq_base), label="baseline")
    # # fig.add_subplot(2, 1, 2)
    # plt.plot(indices, freq_main / np.sum(freq_main), label="main")
    # plt.legend(loc="upper left")

    # plt.savefig("hist_" + args.file_out)

    # # import pdb; pdb.set_trace()

    # # nsamples = (2, 5)
    # # indices = np.random.randint(low=0, high=logits.shape[0], size=10)
    # # features_v = features[indices]
    # # logits_v = logits[indices]

    # # fig=plt.figure(1)
    # # for i in range(1, nsamples[0]*nsamples[1]+1):
    # #     img = images_v[i-1,:,:,:]
    # #     fig.add_subplot(nsamples[0], nsamples[1], i).title.set_text(labels[i])
    # #     plt.imshow(img)
    # # # plt.show()

if __name__ == '__main__':
    main()