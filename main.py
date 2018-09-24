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

from model_cifar10 import cifar10_model_fn
from model_vib_cifar10 import cifar10_vib_model_fn

from input_cifar10 import _cifar10_input_fn

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--dset', type=str, default='cifar10',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--ood_dataset', type=str, default='',
                    help='The identifier for OOD data.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=str, default="50",
                    help= 'The size of the ResNet model to use.'
                          '- if of the format int-int, then wide resnet model is used'
                          '- if of the format int, then resnet is used.')

parser.add_argument('--dim_z', type=int, default=100,
                    help='The dimension of the variaational space Z.')


parser.add_argument('--lamb', type=float, default=1.0,
                    help='The weight of the penalty to pull down logits.')

parser.add_argument('--variant', type=str, default="none",
                    help= 'What variant of the model is to be used.'
                          'none   : plain model without a zero logit'
                          'den    : extra loss is 1 / sum( 1 + exp( wrong logits) )'
                          'num    : extra loss is sum( -1 * exp(wrong label) / normalizer )'
                          'pen    : extra loss is penalty when mean sum of bottom logits is negative'
                          'cen    : extra loss is penalty to center the logits about origin')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=5,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--test', type=int, default=0,
                    help= 'A flag to indicate which test is to be performed '
                          '0: train; main dataset'
                          '1: train; main dataset + ood dataset'
                          '2: test; main dataset'
                          '3: test; ood dataset'
                          '4: test; main dataset + ood dataset'
                          )

parser.add_argument('--ngpus', type=int, default=1,
                    help= 'Number of gpus to be used. 0 -> cpu')


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
    keep_checkpoint_max=100,
    # save_checkpoints_secs=1200,
    # save_checkpoints_steps=1e3,
    session_config=session_config,
    # save_summary_steps=100,
    )

  if FLAGS.variant == "viby":
    model_fn = cifar10_vib_model_fn
  else:
    model_fn = cifar10_model_fn

  classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size' : FLAGS.resnet_size,
          'data_format' : aconfig["data_format"],
          'batch_size'  : FLAGS.batch_size,
          'variant'     : FLAGS.variant,
          'model_dir'   : FLAGS.model_dir,
          'lamb'        : FLAGS.lamb,
          'dim_z'       : FLAGS.dim_z,
          'predict'     : FLAGS.test not in [0, 1],
      },
      )

  input_fn = _cifar10_input_fn
  _hinged_flag = FLAGS.variant != "none"

  if FLAGS.test == 1 and not _hinged_flag:
    print(
      "WARNING: OOD dataset provided but no hinge, are you sure?",
      file=sys.stderr,
      )

  if FLAGS.test in [0, 1]:
    # tensors_to_log = {
    #   'accuracy': 'accuracy'
    # }
    # logging_hook = tf.train.LoggingTensorHook(
    #   tensors=tensors_to_log, 
    #   every_n_iter=100,
    # )

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):

      classifier.train(
        input_fn=lambda: input_fn(FLAGS.test, FLAGS.dset, FLAGS.ood_dataset, FLAGS.batch_size, FLAGS.epochs_per_eval, hinged=_hinged_flag),
        # hooks=[logging_hook],
        )

      # Evaluate the model and print results
      eval_results = classifier.evaluate(
        input_fn=lambda: input_fn(FLAGS.test, FLAGS.dset, FLAGS.ood_dataset, FLAGS.batch_size, is_validating=True, hinged=_hinged_flag),
        # hooks=[logging_hook]
        )

  else:
    logging_hook = tf.train.LoggingTensorHook(
      tensors={
        "accuracy": "accuracy",
      },
      every_n_iter=1,
    )
    eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(FLAGS.test, FLAGS.dset, FLAGS.ood_dataset, FLAGS.batch_size, hinged=_hinged_flag),
      # hooks=[logging_hook]
      )

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
