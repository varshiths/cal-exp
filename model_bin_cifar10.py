
import os

import tensorflow as tf
import tensorflow_probability as tfp

import resnet_model
from utils import custom_cross_entropy, smooth_neg_labels

from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES


# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

def cifar10_bin_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format']
    )

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  clabels = labels[:, :_NUM_CLASSES]
  
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  probs = tf.sigmoid(logits)

  slabels, smask = smooth_neg_labels(clabels, params["cutoff_weight"], params["pen_prob"])
  # slabels, smask = smooth_neg_labels(clabels, 1*1/_NUM_CLASSES, 0.45)

  loss = tf.reduce_mean(custom_cross_entropy(probs, slabels))
  loss = tf.identity(loss, name="loss_vec")
  loss_sum = tf.summary.scalar("loss", loss)

  rate = tf.reduce_max(probs, axis=1, keepdims=True)

  # loss = tf.Print(loss, [smask], summarize=100, message="smask: ")
  # loss = tf.Print(loss, [tf.reduce_mean(probs)], summarize=100, message="mean: ")
  # loss = tf.Print(loss, [rate], summarize=100, message="rate: ")
  # loss = tf.Print(loss, [clabels, slabels], summarize=100, message="slabels: ")

  classes = tf.argmax(logits, axis=1)
  accuracy = tf.metrics.accuracy( tf.argmax(labels, axis=1), classes, name="accuracy_metric")
  accuracy = tf.identity(accuracy[1], name="accuracy_vec")
  accuracy_sum = tf.summary.scalar("accuracy", accuracy)

  if mode == tf.estimator.ModeKeys.EVAL or params["predict"]:

    # print
    print_labels = tf.argmax(labels, 1)
    # print_preds = tf.argmax(logits, 1)
    print_probs = tf.concat([probs, rate], axis=1)
    print_logits = tf.concat([logits, rate], axis=1)

    hooks = [tf.train.SummarySaverHook(
          summary_op=tf.summary.merge([accuracy_sum]),
          output_dir=os.path.join(params["model_dir"], "eval_core"),
          save_steps=1,
          )]

    # # printing stuff if predict
    if params["predict"]:
      loss = tf.Print(loss, [print_labels], summarize=1000000, message='Targets')
      # loss = tf.Print(loss, [print_preds], summarize=1000000, message='Predictions')
      loss = tf.Print(loss, [print_probs], summarize=1000000, message='Probs')
      loss = tf.Print(loss, [print_logits], summarize=1000000, message='Logits')
      hooks = []

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      evaluation_hooks=hooks,
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
        tf.cast(global_step, tf.int32), boundaries, values, name="learning_rate_vec")

    learning_rate_sum = tf.summary.scalar("learning_rate", learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM
        )

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    hook = tf.train.SummarySaverHook(
      summary_op=tf.summary.merge([accuracy_sum, learning_rate_sum]),
      save_steps=1,
      )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[hook],
        )