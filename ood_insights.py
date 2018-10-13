
import numpy as np

def area_under(_points):
    points = _points.copy()
    points.sort(axis=0)
    areas = (points[1:, 0]-points[:-1, 0])*(points[:-1, 1]+points[1:, 1])/2
    return np.sum(areas)

def ROC(probs, targets, step_size=1e-3):

    npts = int(1/step_size)
    points = np.zeros((npts, 2))

    for i in range(npts):
        points[i, 0] = FPR(probs, targets, i*step_size)
        points[i, 1] = TPR(probs, targets, i*step_size)

    return points

def PR(probs, targets, reverse=False, step_size=1e-3):

    if reverse:
        targets = 1-targets
        probs = 1-probs

    npts = int(1/step_size)
    points = np.zeros((npts, 2))

    for i in range(npts):
        points[i, 0] = Recall(probs, targets, i*step_size)
        pr = Precision(probs, targets, i*step_size)
        if pr is None:
            if i > 0:
                points[i, 1] = points[i-1, 1]
            else:
                points[i, 1] = 0

    return points

def Recall(*args):
    return TPR(*args)

def Precision(probs, targets, thresh):
    # assumes targets == 1 is the positive class
    preds = (probs >= thresh).astype(np.float32)
    cmask = (preds == targets).astype(np.float32)

    TP = np.sum((preds == 1).astype(np.float32)*cmask)
    FP = np.sum((preds == 1).astype(np.float32)*(1-cmask))

    if TP + FP == 0:
        print("Warning: No points with probs above", thresh)        
        return None
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

    # function that guides thresh 1-TPR
    def binary_thresh_search(l, r, func, val):
        mid = l+(r-l)/2
        # print(mid, func(mid), val)
        if abs(func(mid) - val) < tolerance:
            return mid
        else:
            try:
                if func(mid) > val:
                    return binary_thresh_search(l, mid, func, val)
                else:
                    return binary_thresh_search(mid, r, func, val)
            except Exception as e:
                print(" Warning:", e)
                print(" Threshold:", mid)
                return mid

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

def distr_rates(rate, imask, name):

    irate = rate[np.where(imask)]
    if np.sum(imask) != 0:
        print("%s: Min, Max, Mean, Std : %.2f, %.2f, %.2f, %.2f " % (name, np.min(irate), np.max(irate), np.mean(irate), np.std(irate)))
    else:
        print("%s: No Samples" % (name))
