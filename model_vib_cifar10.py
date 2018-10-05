
import os

import tensorflow as tf
import tensorflow_probability as tfp

import resnet_model
from utils import custom_softmax_cross_entropy

from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES


_NUM_SAMPLES_Z = 64

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

def cifar10_vib_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  _DIM_Z = params["dim_z"]
  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _DIM_Z*2, params['data_format']
    )
  logits_from_z = tf.layers.Dense(_NUM_CLASSES)

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  clabels = labels[:, :_NUM_CLASSES]
  
  params_z = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  mean_z = params_z[:, :_DIM_Z]
  std_z = tf.nn.softplus(params_z[:, _DIM_Z:])

  # import pdb; pdb.set_trace()

  _mean_z = tf.expand_dims(mean_z, axis=1)
  _std_z = tf.expand_dims(std_z, axis=1)
  distr_z = tfp.distributions.MultivariateNormalDiag(
      loc=_mean_z,
      scale_diag=_std_z,
      # loc=tf.broadcast_to(_mean_z, [-1, _NUM_CLASSES, _DIM_Z]),
      # scale_diag=tf.broadcast_to(_std_z, [-1, _NUM_CLASSES, _DIM_Z]),
    )

  # squeeze the _NUM_CLASSES dim
  samples_z = tf.squeeze(distr_z.sample(_NUM_SAMPLES_Z), axis=2)
  logits_samples = logits_from_z(samples_z)
  br_clabels = tf.expand_dims(clabels, axis=0)
  # br_clabels = tf.broadcast_to(clabels, shape=[_NUM_SAMPLES_Z, -1, _NUM_CLASSES])
  # mean across samples and batch
  cross_entr = tf.reduce_mean(
      tf.reduce_mean(custom_softmax_cross_entropy(logits=logits_samples, labels=br_clabels), axis=0),
      axis=0,
    )

  mean_prior = tf.get_variable("prior_mean", (_NUM_CLASSES, _DIM_Z))
  std_prior = tf.nn.softplus(tf.get_variable("prior_std", (_NUM_CLASSES, _DIM_Z)))

  distr_prior_z = tfp.distributions.MultivariateNormalDiag(
      loc=tf.expand_dims(mean_prior, axis=0),
      scale_diag=tf.expand_dims(std_prior, axis=0),
    )

  kldivs = tfp.distributions.kl_divergence(distr_z, distr_prior_z)
  kldivs_corr = clabels * kldivs
  kldiv_term = tf.reduce_mean(
      tf.reduce_sum(kldivs_corr, axis=1),
      axis=0,
    )

  loss = cross_entr + params["lamb"]*kldiv_term
  loss = tf.identity(loss, name="loss_vec")
  loss_sum = tf.summary.scalar("loss", loss)

  # squeeze the _NUM_CLASSES dim
  sample_z = distr_z.sample()
  logits = logits_from_z(tf.squeeze(sample_z, axis=1))
  probs = tf.nn.softmax(logits, 1)

  # ratio of e and m for the a particular input
  # e_by_m = distr_z.prob(sample_z) / distr_prior_z.prob(sample_z)
  # sum over num classes
  # rate = tf.reduce_sum(e_by_m * 1, axis=1, keepdims=True)
  rate = tf.sigmoid(tf.reduce_sum(kldivs * probs, axis=1, keepdims=True))

  # loss = tf.Print(loss, [sample_z], message="E/M")
  # loss = tf.Print(loss, [rate], message="Rate")
  # loss = tf.Print(loss, [distr_z.prob(sample_z), distr_prior_z.prob(sample_z)], message="E/M")


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
