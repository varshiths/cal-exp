
import tensorflow as tf

EPSILON = 1e-10
INF = 1e10


def custom_softmax_cross_entropy(logits, labels):

  scaled_logits = logits - tf.reduce_max(logits, axis=-1)
  softmax = tf.nn.softmax(logits, axis=-1)

  pdtp = -tf.reduce_sum(labels * tf.log( softmax + EPSILON ), axis=-1)

  return pdtp