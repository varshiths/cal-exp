
import tensorflow as tf

def custom_softmax_cross_entropy(logits, labels):

  scaled_logits = logits - tf.reduce_max(logits, axis=-1)
  softmax = tf.nn.softmax(logits)

  pdtp = -tf.reduce_sum(labels * tf.log( softmax + EPSILON ), axis=1)

  return tf.reduce_mean(pdtp)