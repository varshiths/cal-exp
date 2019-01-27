
import os

import tensorflow as tf
import tensorflow_probability as tfp

import resnet_model
from utils import per_class_bin_loss

from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES
from mmce import per_class_mmce_loss

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

def cifar10_binc_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""

  _DIM_Z = params["dim_z"]
  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _DIM_Z, params['data_format']
    )
  logits_from_z = tf.layers.Dense(_NUM_CLASSES, name="logits_z")
  confs_from_z = tf.layers.Dense(_NUM_CLASSES, use_bias=False, name="confs_z")

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  clabels = labels[:, :_NUM_CLASSES]
  
  z_space = network(inputs, mode == tf.estimator.ModeKeys.TRAIN, name="main")
  # z_space = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  z_space = tf.nn.relu(z_space)

  logits = logits_from_z(z_space)
  probs = tf.nn.softmax(logits, axis=1)
  base_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=clabels), 
    name="base_loss"
    )

  confs = confs_from_z(z_space)
  confs = tf.sigmoid(confs)

  # slabels, smask = smooth_neg_labels(clabels, params["cutoff_weight"], params["pen_prob"])
  # ct_loss = tf.reduce_mean(custom_cross_entropy(confs, slabels))
  ct_loss = tf.reduce_mean(per_class_bin_loss(confs, clabels, params["milden"]), axis=[0, 1])
  # cc_term = tf.reduce_sum(per_class_mmce_loss(confs, clabels), axis=0)
  cc_term = per_class_mmce_loss(confs, clabels)

  loss = base_loss + params["lamb"]*(ct_loss + params["mmcec"]*cc_term)
  loss = tf.identity(loss, name="loss_vec")
  loss_sum = tf.summary.scalar("loss", loss)

  rate = tf.reduce_max(confs, axis=1)
  
  # summaries
  conf_hist_sum = tf.summary.histogram("confidence", confs)

  _bshape = tf.shape(rate)
  # construct binary labels
  blabels = tf.logical_not(tf.equal(
      tf.argmax(labels, axis=1), _NUM_CLASSES,
    ))

  # calibration graph
  cpreds = tf.greater(rate, 0.5)
  crates = tf.where(
      cpreds,
      rate, 
      1-rate,
    )
  cps = tf.to_float(
      tf.logical_and(cpreds, blabels),
    )

  rate_cal = cps*crates
  rate_cal_sum = tf.summary.histogram("accur_confidence", rate_cal)

  # loss = tf.Print(loss, [smask], summarize=100, message="smask: ")
  # loss = tf.Print(loss, [tf.reduce_mean(probs)], summarize=100, message="mean: ")
  # loss = tf.Print(loss, [rate], summarize=100, message="rate: ")
  # loss = tf.Print(loss, [clabels, slabels], summarize=100, message="slabels: ")

  classes = tf.argmax(logits, axis=1)
  accuracy_m = tf.metrics.accuracy( tf.argmax(clabels, axis=1), classes, name="accuracy_metric")
  accuracy = tf.identity(accuracy_m[1], name="accuracy_vec")
  accuracy_sum = tf.summary.scalar("accuracy", accuracy)

  if mode == tf.estimator.ModeKeys.EVAL or params["predict"]:

    # print # note this is labels not clabels
    print_labels = tf.argmax(labels, axis=1)
    print_rate = rate
    print_confs = confs
    print_probs = probs
    print_logits = logits

    # hooks = [tf.train.SummarySaverHook(
    #       summary_op=tf.summary.merge([accuracy_sum]),
    #       output_dir=os.path.join(params["model_dir"], "eval_core"),
    #       save_steps=1,
    #       )]
    hooks = []
    eval_metric_ops = { "accuracy": accuracy_m }

    # # printing stuff if predict
    if params["predict"]:
      loss = tf.Print(loss, [print_labels], summarize=1000000, message='Targets')
      loss = tf.Print(loss, [print_rate], summarize=1000000, message='Rate')
      loss = tf.Print(loss, [print_confs], summarize=1000000, message='Confs')
      loss = tf.Print(loss, [print_probs], summarize=1000000, message='Probs')
      loss = tf.Print(loss, [print_logits], summarize=1000000, message='Logits')
      hooks = []
      eval_metric_ops = {}

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops = eval_metric_ops
      # evaluation_hooks=hooks,
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
      summary_op=tf.summary.merge([accuracy_sum, learning_rate_sum, conf_hist_sum, rate_cal_sum]),
      save_steps=1,
      )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[hook],
        )
