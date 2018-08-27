# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Authors: IITB
# Modified to run calibration experiments.
# ==============================================================================
"""Runs a ResNet/WideResNet model on the CIFAR-10 and some other datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from warnings import warn

import tensorflow as tf

import resnet_model
from mmce import *

from cifar10 import record_dataset, get_filenames, parse_record, preprocess_image
from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES

from ood import _NUM_IMAGES_OOD, get_filenames as get_ood_filenames

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--ood_dataset', type=str, default='',
                    help='The identifier for OOD data.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=str, default="50",
                    help= 'The size of the ResNet model to use.'
                          '- if of the format int-int, then wide resnet model is used'
                          '- if of the format int, then resnet is used.')

parser.add_argument('--hinged', type=bool, default=False,
                    help= 'If an additional zero logit is to be added to the output'
                          'of te model.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--test', type=int, default=0,
                    help= 'A flag to indicate which test is to be performed '
                          '0: No test. Proceed with training'
                          '1: Evaluate on eval set'
                          )

parser.add_argument('--ngpus', type=int, default=1,
                    help= 'Number of gpus to be used. 0 -> cpu')

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_TRAIN_VAL_SPLIT_SEED = 0

'''
def get_filenames(mode, data_dir, ood_dataset):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if mode == 0:
    datafile = []
    # datafile = [os.path.join(data_dir, ood_dataset + '_ood_batch.bin')]
    # assert os.path.exists(datafile), (
        # 'Run make_ood_data.py for train purposes first and copy it into the cifar-10-batches-bin directory.' )
    return [
      os.path.join(data_dir, 'data_batch_%d.bin' % i)
      for i in range(1, _NUM_DATA_FILES + 1)
    ] + datafile

  elif mode == 1:
    datafile = os.path.join(data_dir, ood_dataset + '_ood_test_batch.bin')
    assert os.path.exists(datafile), (
        'Run make_ood_data.py for test purposes first and copy it into the cifar-10-batches-bin directory.' )
    return [
      os.path.join(data_dir, 'test_batch.bin')
    ] + [datafile]

  elif mode == 2:
    datafile = os.path.join(data_dir, ood_dataset + '_ood_test_batch.bin')
    assert os.path.exists(datafile), (
        'Run make_ood_data.py for test purposes first and copy it into the cifar-10-batches-bin directory.' )
    # return [datafile]
'''

def get_train_or_val(dataset, NDICT, is_validating):
  # first randomly shuffle the exact same way using the same constant seed
  dataset = dataset.shuffle(
    buffer_size=NDICT['validation'] + NDICT['train'],
    seed=_TRAIN_VAL_SPLIT_SEED,
    )
  # pick subset based on whether you're validating or training
  if not is_validating:
    dataset = dataset.take(NDICT["train"])
  else:
    dataset = dataset.skip(NDICT["train"])
  flag = int(is_validating); size = NDICT["validation"]*flag + NDICT["train"]*(1-flag)

  return dataset, size

def input_fn(mode, data_dir, ood_dataset, batch_size, num_epochs=1, is_validating=False, hinged=False):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    mode: An int denoting whether the input is for training, test with the corresponding datasets.
      0: train; main dataset
      1: train; main dataset + ood dataset
      2: test; main dataset
      3: test; ood dataset
      4: test; main dataset + ood dataset
    data_dir: The directory containing the input data.
    ood_dataset: String pointing to a particular ood dataset.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  is_training = mode in [0, 1]
  is_ood = mode in [1, 3, 4]
  is_main = mode in [0, 1, 2, 4]

  assert ( is_training or not is_validating ), ("Can't perform test and validation")
  assert (is_ood or is_main), ("Select at least one dataset.")
  
  filenames = []
  ds_size, ods_size = 0, 0
  if is_main:
    filename = get_filenames(0 if is_training else 1, data_dir)
    filenames += filename
    dataset = record_dataset(filename)
    ds_size = _NUM_IMAGES["test"]
    if is_training:
      dataset, ds_size = get_train_or_val(dataset, _NUM_IMAGES, is_validating)
  if is_ood:
    filename = get_ood_filenames(0 if is_training else 1, data_dir, ood_dataset)
    ood_dataset = record_dataset(filename)
    filenames += filename
    ods_size = _NUM_IMAGES_OOD["test"]
    if is_training:
      ood_dataset, ods_size = get_train_or_val(ood_dataset, _NUM_IMAGES, is_validating)

  # print("------------------------------")
  # print("filenames: ", filenames)
  # print("------------------------------")

  # merge the two datasets after train val split
  if not is_main:
    dataset = ood_dataset
    ds_size = ods_size
  elif is_ood:
    dataset = dataset.concatenate(ood_dataset)
    ds_size += ods_size

  if is_training and not is_validating:
    dataset = dataset.shuffle(
        buffer_size=ds_size
      )
  dataset = dataset.map(lambda x: parse_record(x, _NUM_CLASSES + int(hinged)))
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training and not is_validating), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format']
    )

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

  # adding logit 0 for NOTA
  if params["hinged"]:
    logits = tf.pad( logits, [[0,0],[0, 1]], "CONSTANT")

  classes = tf.argmax(logits, axis=1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  accuracy = tf.metrics.accuracy( tf.argmax(labels, axis=1), classes)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
  loss = tf.identity(loss, name="loss")

  if mode == tf.estimator.ModeKeys.EVAL:

    # printing stuff
    loss = tf.Print(loss, [tf.argmax(labels, 1)], summarize=1000000, message='Targets')
    loss = tf.Print(loss, [tf.argmax(logits, 1)], summarize=1000000, message='Predictions')
    loss = tf.Print(loss, [tf.nn.softmax(logits)], summarize=1000000, message='Probs')
    loss = tf.Print(loss, [logits], summarize=1000000, message='Logits')

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={'accuracy': accuracy},
        )

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create tensors for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.identity(accuracy[1], name='train_accuracy')

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM
        )

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        )


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  aconfig = {
    "data_format": "channels_last",
  }
  if FLAGS.ngpus == 1:
    FLAGS.data_format = "channels_first"

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)
  session_config = tf.ConfigProto(gpu_options=gpu_options)
  session_config.gpu_options.allow_growth = True
  
  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig(
    save_checkpoints_secs=1e9,
    session_config=session_config,
    )

  classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size' : FLAGS.resnet_size,
          'data_format' : aconfig["data_format"],
          'batch_size'  : FLAGS.batch_size,
          'hinged'      : FLAGS.hinged,
      })

  if FLAGS.test == 1 and not FLAGS.hinged:
    print(
      "WARNING: OOD dataset provided but no hinge, are you sure?",
      file=sys.stderr,
      )

  if FLAGS.test in [0, 1]:
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      tensors_to_log = {
        'learning_rate': 'learning_rate',
        'train_accuracy': 'train_accuracy'
        # 'loss': 'loss',
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      classifier.train(
          input_fn=lambda: input_fn(FLAGS.test, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size, FLAGS.epochs_per_eval, hinged=FLAGS.hinged),
          hooks=[logging_hook],
          )

      # Evaluate the model and print results
      eval_results = classifier.evaluate(
          input_fn=lambda: input_fn(FLAGS.test, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size, is_validating=True, hinged=FLAGS.hinged))
      # print(eval_results)

  else:
    eval_results = classifier.evaluate(
        input_fn=lambda: input_fn(FLAGS.test, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size, hinged=FLAGS.hinged))
    # print(eval_results)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
