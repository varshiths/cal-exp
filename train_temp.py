#!/usr/bin/python3

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from insights_utils import transform_line 
from insights_utils import get_intervals, get_acc_bucket

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=2000,
                    help='The number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=10,
                    help='The number of images per batch.')

NCLASSES = 10
_NCLASSES = NCLASSES + 1

tensors_to_get = ["Targets[", "Logits["]

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
    logits = tsrsd[tensors_to_get[1]]

    # perform the required reshapes
    targets = targets
    logits = np.reshape(logits, [-1, _NCLASSES])

    assert targets.shape[0] == logits.shape[0], "corrupt: %s" % filename
    return targets, logits

class TemperatureTrainer:

  def __init__(self):

    self.logit = tf.placeholder(tf.float32, [None, NCLASSES])
    self.target = tf.placeholder(tf.int32, [None])
    self.T = tf.Variable(0.93)

    self.loss = self.calculate_loss()

    opt = tf.train.AdagradOptimizer(5e-2)
    self.train_opt = opt.minimize(self.loss)

  def calculate_loss(self):

    slogits = self.logit * self.T
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=slogits)
    return tf.reduce_mean(loss)

  def shuffle(dtargets, dlogits):

    perm = np.random.permutation(dtargets.shape[0])
    dtargets =  np.take(dtargets, perm, axis=0)
    dlogits = np.take(dlogits, perm, axis=0)
    return dtargets, dlogits

  def train(self, dtargets, dlogits):

    bs = FLAGS.batch_size
    dtargets, dlogits = TemperatureTrainer.shuffle(dtargets, dlogits)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(FLAGS.train_epochs):
      for j in range(FLAGS.train_epochs // bs):
        feed_dict = {}
        feed_dict[self.logit] = dlogits[j*bs:(j+1)*bs, :]
        feed_dict[self.target] = dtargets[j*bs:(j+1)*bs]
        val_loss, val_T, _ = sess.run([self.loss, self.T, self.train_opt], feed_dict=feed_dict)
        print ('Loss, Temperature: ', val_loss, val_T)

def main(unused_argv):

  dtargets, dlogits = get_tensors_from_file(FLAGS.file)
  dlogits = dlogits[:, :NCLASSES]

  model = TemperatureTrainer()
  model.train(dtargets, dlogits)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
