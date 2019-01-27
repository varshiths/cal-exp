
import tensorflow as tf

EPSILON = 1e-10
INF = 1e10


def custom_cross_entropy(probs, labels):

  pdtp = -tf.reduce_sum(labels * tf.log( probs + EPSILON ), axis=-1)
  return pdtp

def smooth_neg_labels(clabels, alpha, prob):

  smask = tf.cast(
    tf.random_uniform(tf.shape(clabels), minval=0, maxval=1) < prob,
    dtype=tf.float32,
    )

  # weight = 1
  # weight = alpha
  weight =  tf.clip_by_value(
      1/tf.reduce_sum(smask, axis=1, keepdims=True), 
      alpha,
      INF,
    )

  slabels = clabels - (1-clabels)*smask*weight
  return slabels, smask

def per_class_bin_loss(probs, labels, milden=0.1):

  probs = tf.stack([1-probs, probs], axis=2)
  labels = tf.one_hot(tf.to_int32(labels), depth=2)

  # milden the labels for Class 0 : 1 -> 0.1
  labels = tf.stack([labels[:, :, 0]*milden, labels[:, :, 1]], axis=2)
  # labels = tf.Print(labels, [labels], summarize=1000, message="SLabels")

  entr = custom_cross_entropy(probs, labels)
  return entr
