
import numpy as np

IEPSILON = 0.01
EPSILON = 1e-10
INF = 1e10

def get_intervals(N):
    int0 = np.arange(N)
    int1 = int0+1
    ints = np.stack([int0, int1], axis=1)/N

    # shifting the ends slightly to add extremes
    ints[0][0] += -IEPSILON
    ints[-1][1] += IEPSILON

    return ints, int0

def get_acc_bucket(probs, preds, tgts, confint):
    # mask = (confint[0] <= probs and probs < confint[1]).astype(int)
    mask = np.logical_and(confint[0] <= probs, probs < confint[1]).astype(int)
    if np.sum(mask) == 0:
        accuracy = None
    else:
        accuracy = np.sum((tgts == preds).astype(float)*mask) / np.sum(mask)
    return accuracy, np.sum(mask)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    maxx = np.max(x, axis=axis, keepdims=True)
    emx = np.exp(x - maxx)
    return emx/np.sum(emx, axis=axis, keepdims=True)

def get_tensors_from_file(filename, tensors_to_get, nclasses, num_non_reshape=1):

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

    anss = [ tsrsd[x] for x in tensors_to_get ]
    # targets = tsrsd[tensors_to_get[0]]
    # probs = tsrsd[tensors_to_get[1]]
    # logits = tsrsd[tensors_to_get[2]]

    # perform the required reshapes
    # usually targets

    for i in range(num_non_reshape, len(anss)):
        anss[i] = np.reshape(anss[i], [-1, nclasses])

    assert anss[0].shape[0] == anss[1].shape[0], "corrupt: %s" % filename
    return anss
