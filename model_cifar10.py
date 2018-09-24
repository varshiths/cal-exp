
import os

import tensorflow as tf
import tensorflow_probability as tfp

import resnet_model
from utils import custom_softmax_cross_entropy

from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES


_NUM_PEN_CLASSES = _NUM_CLASSES // 2

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

EPSILON = 1e-10
INF = 1e10


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format']
    )

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

  # adding logit 0 for NOTA
  if params["variant"] != "none":
    logits = tf.pad( logits, [[0,0],[0, 1]], "CONSTANT")

  classes = tf.argmax(logits, axis=1)

  # if mode == tf.estimator.ModeKeys.PREDICT:
  #   predictions = {
  #       'classes': classes,
  #       'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
  #   }
  #   return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  accuracy = tf.metrics.accuracy( tf.argmax(labels, axis=1), classes, name="accuracy_metric")
  accuracy = tf.identity(accuracy[1], name="accuracy_vec")
  accuracy_sum = tf.summary.scalar("accuracy", accuracy)

  # Calculate loss, which includes softmax cross entropy
  base_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), 
    name="base_loss"
    )

  lnfactor = 0
  pvals = 1-labels; pvals = pvals/tf.reduce_sum(pvals, axis=-1, keepdims=True)
  distr = tfp.distributions.Categorical(probs=pvals)
  neg_samples = tf.transpose(distr.sample( [_NUM_PEN_CLASSES] ))
  mask = tf.one_hot(neg_samples, depth=_NUM_CLASSES+1, axis=1)
  mask = tf.reduce_sum(mask, axis=(-1))

  if params["variant"] == "den":
    lnfactor = tf.reduce_mean(tf.log( 1 + tf.reduce_sum( tf.exp(logits) * mask, axis=1 ) ))

  elif params["variant"] == "num":
    lnfactor = -custom_softmax_cross_entropy(logits=logits, labels=mask) / _NUM_PEN_CLASSES

  elif params["variant"] == "pen":
    neg_logits_mean = tf.reduce_mean(mask * logits, axis=1)
    lnfactor = tf.reduce_mean(tf.square(
        tf.nn.softplus( -neg_logits_mean )
      )) / (_NUM_PEN_CLASSES * 20)

  elif params["variant"] == "cen":

    std = tf.get_variable(name="std_logits", shape=(1), initializer=tf.ones_initializer(), trainable=True)
    distr = tfp.distributions.MultivariateNormalDiag(
      loc=tf.zeros(_NUM_CLASSES+1),
      scale_identity_multiplier=std,
      )
    probs = distr.prob(logits)
    lnfactor = -tf.reduce_mean(tf.log(probs + EPSILON))

  loss = base_loss + params["lamb"] * lnfactor
  loss = tf.identity(loss, name="loss_vec")
  loss_sum = tf.summary.scalar("loss", loss)

  if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:

    # # printing stuff if predict
    if mode == tf.estimator.ModeKeys.PREDICT or params["predict"]:
      loss = tf.Print(loss, [tf.argmax(labels, 1)], summarize=1000000, message='Targets')
      loss = tf.Print(loss, [tf.argmax(logits, 1)], summarize=1000000, message='Predictions')
      loss = tf.Print(loss, [tf.nn.softmax(logits)], summarize=1000000, message='Probs')
      loss = tf.Print(loss, [logits], summarize=1000000, message='Logits')

    hook = tf.train.SummarySaverHook(
      summary_op=tf.summary.merge([accuracy_sum]),
      output_dir=os.path.join(params["model_dir"], "eval_core"),
      save_steps=1,
      )
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      evaluation_hooks=[hook],
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
