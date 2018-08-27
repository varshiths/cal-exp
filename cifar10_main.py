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
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import resnet_model
from mmce import *

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

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10 + 1
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


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
    return [datafile]

def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(mode, data_dir, ood_dataset, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    mode: An int denoting whether the input is for training, eval, or specialeval.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(mode, data_dir, ood_dataset))

  if mode == 0:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, mode == 0), label))

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

  # tf.summary.image('images', features, max_outputs=6)

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES-1, params['data_format']
    )

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  # adding logit 0 for NOTA
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

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)
  session_config = tf.ConfigProto(gpu_options=gpu_options)
  session_config.gpu_options.allow_growth = True
  
  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig(
    save_checkpoints_secs=1e9,
    session_config=session_config,
    )

  cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  if FLAGS.test == 0:
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      tensors_to_log = {
        'learning_rate': 'learning_rate',
        'train_accuracy': 'train_accuracy'
        # 'loss': 'loss',
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      cifar_classifier.train(
          input_fn=lambda: input_fn(0, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size, FLAGS.epochs_per_eval),
          hooks=[logging_hook],
          )

      # Evaluate the model and print results
      eval_results = cifar_classifier.evaluate(
          input_fn=lambda: input_fn(1, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size))
      print(eval_results)

  # elif FLAGS.test == 1 or FLAGS.test == 2:
  else:
    eval_results = cifar_classifier.evaluate(
        input_fn=lambda: input_fn(FLAGS.test, FLAGS.data_dir, FLAGS.ood_dataset, FLAGS.batch_size))
    print(eval_results)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
