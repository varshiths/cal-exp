
import tensorflow as tf

EPSILON = 1e-10
INF = 1e10


def custom_softmax_cross_entropy(logits, labels):

  scaled_logits = logits - tf.reduce_max(logits, axis=-1)
  softmax = tf.nn.softmax(logits, axis=-1)

  pdtp = -tf.reduce_sum(labels * tf.log( softmax + EPSILON ), axis=-1)

  return pdtp

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
