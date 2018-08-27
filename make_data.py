#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
# import cv2

import matplotlib.pyplot as plt

# from scipy import ndimage

# from skimage.tranform import resize
# from scipy.misc

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_N = 10000
_NUM_CLASSES = 11
_OOD_CLASS = 10


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--target_data_dir', type=str, default='cifar10_data/cifar-10-batches-bin',
                    help='The path of the data directory to write the binary data to')
parser.add_argument('--source_data_dir', type=str, default='extra_data',
                    help='The path to the data directory')
parser.add_argument('--viz', type=bool, default=False,
                    help='Should I visulalize')
parser.add_argument('--type', type=int, default=0,
                    help='Type of extra data to generate'
                    '0: Noise'
                    '1: Mixed Averaged cifar10'
                    '2: TinyImageNet'
                    '3: Get All Mixed'
                    )

def bin_label(label):
    assert label.dtype == np.uint8, "ensure your labels are uint8"
    return bytes([label])

def bin_image(image):
    # transpose image in [height, width, depth] to [depth, height, width]
    # ensure image is uint8 already
    assert image.dtype == np.uint8, "ensure your images are uint8"
    image = np.transpose(image.astype(np.uint8), (2, 0, 1))
    return bytes(image)

def get_random_train_test(args):
    split = int(0.6 * _N)
    images = (np.random.uniform(size=(_N, _HEIGHT, _WIDTH, _DEPTH))*256).astype(np.uint8)
    labels = (np.ones(shape=(_N))*_OOD_CLASS).astype(np.uint8)
    return images[:split], labels[:split], images[split:], labels[split:]

def get_random_mixed_cifar10(args):

    dfile = os.path.join(args.source_data_dir, 'cifar-10-batches-bin')
    dfile = os.path.join(dfile, 'test_batch.bin')
    data = np.fromfile(dfile, dtype=np.uint8).astype(np.float32)
    data = np.reshape(data, [-1, 1+_HEIGHT*_WIDTH*_DEPTH])

    assert data.shape[0] == _N
    labels = np.ones(shape=_N)*_OOD_CLASS
    images = np.transpose(np.reshape(data[:, 1:], (-1, _DEPTH, _HEIGHT, _WIDTH)), (0,2,3,1))

    images_c = images.copy()
    np.random.shuffle(images_c)
    images = 0.5 * images + 0.5 * images_c

    images, labels = images.astype(np.uint8), labels.astype(np.uint8)
    split = int(0.6 * _N)
    return images[:split], labels[:split], images[split:], labels[split:]

def get_tin(args, _all=True):

    directory = os.path.join(args.source_data_dir, 'tin')
    files = os.listdir(directory) if _all else os.listdir(directory)[:2500]

    images = []
    for i, file in enumerate(files):
        print(i, file)
        file = os.path.join(directory, file)
        image = plt.imread(file)

        x, y = np.random.randint(0, 32, 2)

        if len(image.shape) == 2:
            image = np.stack( [image]*3, axis=2 )

        image = image[x:x+_HEIGHT,y:y+_WIDTH,:]
        images.append(image)
    images = np.stack(images)

    assert images.shape[0] == _N or not _all
    labels = np.ones(shape=_N) * _OOD_CLASS

    images, labels = images.astype(np.uint8), labels.astype(np.uint8)
    split = int(0.6 * _N)
    return images[:split], labels[:split], images[split:], labels[split:]

def visualize(images, labels):
    nsamples = (2, 5)
    indices = np.random.randint(low=0, high=nsamples[0]*nsamples[1], size=10)
    images_v = images[indices]
    labels_v = labels[indices]

    fig=plt.figure(1, figsize=(_HEIGHT, _WIDTH))
    for i in range(1, nsamples[0]*nsamples[1]+1):
        img = images_v[i-1,:,:,:]
        fig.add_subplot(nsamples[0], nsamples[1], i).title.set_text(labels[i-1])
        plt.imshow(img)
    plt.show()

def main(args):

    pref = ""
    if args.type == 0:
        pref = "noise"
        images, labels, images_t, labels_t = get_random_train_test(args)
    elif args.type == 1:
        pref = "cifar10mix"
        images, labels, images_t, labels_t = get_random_mixed_cifar10(args)
    elif args.type == 2:
        pref = "tin"
        images, labels, images_t, labels_t = get_tin(args)
    elif args.type == 3:
        pref = "all"
        # images, labels, images_t, labels_t = get_all(args)

    if args.viz:
        visualize(images, labels)

    _filename_test = pref + "_ood_test_batch.bin"
    _filename = pref + "_ood_batch.bin"

    with open(os.path.join(args.target_data_dir, _filename), "wb") as file:
        for i, image in enumerate(images):
            file.write(bin_label(labels[i]))
            file.write(bin_image(image))

    with open(os.path.join(args.target_data_dir, _filename_test), "wb") as file:
        for i, image_t in enumerate(images_t):
            file.write(bin_label(labels_t[i]))
            file.write(bin_image(image_t))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
