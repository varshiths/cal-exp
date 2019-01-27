
import tensorflow as tf

INF=1e10
SINF=100


def pair_tensor_class_wise(tsr):
  '''
  Input:
    one tensors of shape
    nclasses x batch
  Returns:
    one tensors of shape
    nclasses x batch*batch x 2
  '''
  _nc = tf.shape(tsr)[0]
  _bs = tf.shape(tsr)[1]

  # nclasses x batch x 1
  extsr = tf.expand_dims(tsr, axis=2)
  # tile entire batch batch times
  bbtsr = tf.tile(extsr, [1, 1, _bs])
  # tile each element batch times
  ebtsr = tf.transpose(bbtsr, [0, 2, 1])

  pairs_tsr = tf.concat(
    [
      tf.reshape(bbtsr, [_nc, _bs*_bs, 1]),
      tf.reshape(ebtsr, [_nc, _bs*_bs, 1]),
    ], axis=2)

  return pairs_tsr

def per_class_mmce_loss_v0(confs, labels):
  '''
  returns a tensor of shape
    nclasses
  '''

  _mshape = tf.shape(labels)
  _nclasses = _mshape[1]

  cmask = tf.where(
      tf.equal(tf.to_float(
          tf.greater(confs, 0.5)
        ), labels),
      tf.ones(_mshape), 
      tf.zeros(_mshape),
    )
  # do not let gradient pass through argmax step
  cmask = tf.stop_gradient(cmask)

  # reshape from batch x nclasses -> nclasses x batch
  confs = tf.transpose(confs)
  cmask = tf.transpose(cmask)

  n = tf.to_float(_mshape[0])
  mv = tf.clip_by_value(tf.reduce_sum(cmask, axis=1), 1, n-1)
  
  pairs_confs = pair_tensor_class_wise(confs)
  pairs_cmask = pair_tensor_class_wise(cmask)

  confs_kernel = mmce_kernel(pairs_confs[:, :, 0], pairs_confs[:, :, 1])

  scmask = tf.reduce_sum(pairs_cmask, axis=2)
  ic_ic_mask = tf.to_float(tf.equal(scmask, 0))
  c_c_mask = tf.to_float(tf.equal(scmask, 2))

  c_ic_mask = tf.to_float(tf.equal(pairs_cmask[:, :, 0], 1)) * \
              tf.to_float(tf.equal(pairs_cmask[:, :, 1], 0))

  ls = tf.reduce_sum(ic_ic_mask*pairs_confs[:, :, 0]*pairs_confs[:, :, 1]*confs_kernel, axis=1) / tf.square(mv-n)
  ls += tf.reduce_sum(c_c_mask*(1.0-pairs_confs[:, :, 0])*(1.0-pairs_confs[:, :, 1])*confs_kernel, axis=1) / tf.square(n)
  ls += -2*tf.reduce_sum(c_ic_mask*(1.0-pairs_confs[:, :, 0])*pairs_confs[:, :, 1]*confs_kernel, axis=1) / ((mv-n)*n)

  # ls = tf.Print(ls, [pairs_confs], message="pairs_confs: ", summarize=SINF)
  # ls = tf.Print(ls, [pairs_cmask], message="pairs_cmask: ", summarize=SINF)
  # ls = tf.Print(ls, [mv], message="mv: ", summarize=SINF)
  # ls = tf.Print(ls, [n], message="n: ", summarize=SINF)

  return tf.sqrt(ls + 1e-10)

def per_class_mmce_loss(confs, labels):
  '''
  labels: batch x classes (already one hot)
  confs: batch x classes
  '''
  
  confs = tf.reshape(confs, [-1])
  labels_corr = tf.reshape(labels, [-1])

  pmask = tf.greater(confs, 0.5)
  labels_pred = tf.to_float(pmask)
  confs_pred = tf.where(
      pmask,
      confs,
      1-confs
    )

  return calibration_mmd_loss(confs_pred, labels_pred, labels_corr)

def mmce_kernel(p1, p2):
  return tf.exp(-1.0*tf.abs(p1-p2)/(2*0.2))

def tf_kernel(matrix):
  return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  #+ tf.exp(-1.0*tf.square(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2*0.2))

def calibration_mmd_loss(predicted_probs, predicted_labels, correct_labels):
  # predicted_probs = tf.nn.softmax(logitsx)
  # predicted_labels = tf.argmax(predicted_probs, axis=1)
  # predicted_probs = tf.reduce_max(predicted_probs, 1)

  correct_mask = tf.where(tf.equal(correct_labels, predicted_labels), tf.ones(tf.shape(correct_labels)), tf.zeros(tf.shape(correct_labels)))

  k = tf.to_int32(tf.reduce_sum(correct_mask))
  k_p = tf.to_int32(tf.reduce_sum(1.0 - correct_mask))
  cond_k = tf.where(tf.equal(k, 0), 0, 1)
  cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)
  k = tf.maximum(k, 1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
  k_p = tf.maximum(k_p, 1)*cond_k_p*cond_k + (1 - cond_k_p*cond_k)*(tf.shape(correct_mask)[0] - 2)
  
  correct_prob, _ = tf.nn.top_k(predicted_probs*correct_mask, k)
  incorrect_prob, _ = tf.nn.top_k(predicted_probs*(1 - correct_mask), k_p)

  def get_pairs(tensor1, tensor2):
    # print (tensor1)
    correct_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1), [1, tf.shape(tensor1)[0]]), 2)
    incorrect_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor2)[0]]), 2)

    correct_prob_pairs = tf.concat([correct_prob_tiled, tf.transpose(correct_prob_tiled, [1, 0, 2])], axis=2)
    incorrect_prob_pairs = tf.concat([incorrect_prob_tiled, tf.transpose(incorrect_prob_tiled, [1, 0, 2])], axis=2)

    correct_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1), [1, tf.shape(tensor2)[0]]), 2)
    incorrect_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor1)[0]]), 2)

    correct_incorrect_pairs = tf.concat([correct_prob_tiled_1, tf.transpose(incorrect_prob_tiled_1, [1, 0, 2])], axis=2)
    return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs
  
  correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)

  correct_kernel = tf_kernel(correct_prob_pairs)
  incorrect_kernel = tf_kernel(incorrect_prob_pairs)
  correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs)
  
  sampling_weights_correct = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.transpose(tf.expand_dims(1.0 - correct_prob, 1)))
  sampling_weights_incorrect = tf.matmul(tf.expand_dims(incorrect_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1)))
  
  sampling_correct_incorrect = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1)))
  correct_denom = tf.reduce_sum(1.0 - correct_prob)
  incorrect_denom = tf.reduce_sum(incorrect_prob)

  total = tf.reduce_sum(correct_mask) + tf.reduce_sum(1.0 - correct_mask)
  m = tf.reduce_sum(correct_mask)
  n = tf.reduce_sum(1.0 - correct_mask)

  mmd_error = 1.0/(m*m + 1e-5)  * tf.reduce_mean(correct_kernel*sampling_weights_correct)
  mmd_error += 1.0/(n*n + 1e-5) * tf.reduce_mean(incorrect_kernel*sampling_weights_incorrect)
  mmd_error -= 2.0/(m*n + 1e-5) * tf.reduce_mean(sampling_correct_incorrect* correct_incorrect_kernel)
  
  return tf.maximum(tf.stop_gradient(tf.to_float(cond_k*cond_k_p))*tf.sqrt(mmd_error + 1e-10), 0.0)

def get_median_value(v):
  v = tf.reshape(v, [-1])
  m = tf.shape(v)[0]//2
  m = tf.Print (m, [m], message='m_value', summarize=100)
  return tf.nn.top_k(v, m).values[m-1]

def calibration_kernel_regression_loss(logits, correct_labels):
  predicted_probs = tf.nn.softmax(logits)
  #predicted_probs = tf.Print(predicted_probs, [tf.shape(predicted_probs)], summarize=100, message='Pred Probs')                                                                                                                              
  predicted_labels = tf.argmax(predicted_probs, axis=1)

  correct_mask = tf.where(tf.equal(correct_labels, predicted_labels), tf.ones(tf.shape(correct_labels)), tf.zeros(tf.shape(correct_labels)))
  sigma = 1.0
  print ('Corect_mask: ', correct_mask)

  def tf_kernel(matrix):
    retval = tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.05)) # * (matrix[:, :, 0] - tf.stop_gradient(matrix[:, :, 1]))/(2*0.05*0.05))
    retval += tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.1)) # * (matrix[:, :, 0] - tf.stop_gradient(matrix[:, :, 1]))/(2*0.1*0.1))
    retval += tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2)) # * (matrix[:, :, 0] - tf.stop_gradient(matrix[:, :, 1]))/(2*0.2*0.2))
    retval /= 2.0
    return retval

  top_token_probs = tf.reduce_max(predicted_probs, 1)
  #top_token_probs = tf.Print(top_token_probs, [tf.shape(top_token_probs)], summarize=100, message='top token')                                                                                                                               
  top_tokens_tiled = tf.tile(tf.expand_dims(top_token_probs, 1), [1, tf.shape(top_token_probs)[0]])
  element_pairs = tf.concat([tf.expand_dims(top_tokens_tiled, 2), tf.expand_dims(tf.transpose(top_tokens_tiled), 2)], axis=2)
  #element_pairs = tf.Print(element_pairs, [tf.shape(element_pairs)], summarize=100, message='element pairs')                                                                                                                                 
  kernel_matrix = tf_kernel(element_pairs)
  correct_mask_transpose = tf.transpose(tf.expand_dims(correct_mask, 1)) # 1 X BATCH                                                                                                                                                          
  numer = tf.reduce_sum(kernel_matrix*correct_mask_transpose, 1)
  #numer = tf.Print(numer, [tf.shape(kernel_matrix), tf.shape(numer)], summarize=100, message='Kernel matrix')                                                                                                                                
  denom = tf.reduce_sum(kernel_matrix, 1)
  #mmd_error = tf.reduce_sum(1.0*numer/(denom + 1e-10) - top_token_probs)                                                                                                                                                                     
  #denom = tf.Print(denom, [numer, denom], summarize=100, message='mmd_error')                                                                                                                                                                
  mmd_error = tf.reduce_mean(tf.abs(1.0*numer/(denom + 1e-10) - top_token_probs))
  return mmd_error
