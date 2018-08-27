import tensorflow as tf

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

def get_median_value(v):
  v = tf.reshape(v, [-1])
  m = tf.shape(v)[0]//2
  m = tf.Print (m, [m], message='m_value', summarize=100)
  return tf.nn.top_k(v, m).values[m-1]

def sample_many_tokens(logits, target_output, num_samples):
  '''Sample many tokens for expectation '''
  samples = tf.multinomial(logits, num_samples)
  index_range = tf.range(0, tf.shape(logits)[0], 1)
  index_range = tf.expand_dims(index_range, 1)
  index_range_tiled = tf.tile(index_range, [1, num_samples])
  index_range_tiled_e = tf.expand_dims(index_range_tiled, 2)
  samples_e = tf.expand_dims(samples, 2)
  gather_concat = tf.concat([tf.to_int64(index_range_tiled_e), tf.to_int64(samples_e)], 2)
  sampled_probs = tf.gather_nd(logits, gather_concat)

  sampled_probs = tf.reshape(sampled_probs, [-1])
  
  target_output_e = tf.expand_dims(target_output, 1)
  target_output_tiled = tf.tile(target_output_e, [1, num_samples])
  target_outputs_reshaped = tf.reshape(target_output_tiled, [-1])

  '''target_weight_e = tf.expand_dims(target_weight, 1)
  target_weight_tiled = tf.tile(target_weight_e, [1, num])
  target_weight_reshaped = tf.reshape(target_weight_tiled, [-1])'''
  
  return sampled_probs, tf.reshape(samples, [-1]),  target_outputs_reshaped


def calibration_mmd_loss_new(logits,  correct_labels, is_topk=False, samples_predicted=None):
    # predicted_probs = tf.nn.softmax(logits)
    # range_index = tf.to_int64(tf.expand_dims(tf.range(0, tf.shape(predicted_probs)[0]), 1))
    # predicted_labels = tf.argmax(predicted_probs, axis=1)
    predicted_labels = samples_predicted

    # predicted_probs = tf.Print(predicted_probs, [predicted_labels], summarize=50, message='Predicted_probs')
    
    # gather_index = tf.concat([range_index, tf.expand_dims(predicted_labels, 1)], axis=1)
    predicted_probs = logits

    # predicted_labels = tf.Print(predicted_labels, [tf.shape(predicted_labels), tf.shape(correct_labels)], summarize=100, message='Predicted')

    if is_topk:
      predicted_probs = tf.nn.softmax(logits)
      predicted_topk, label_indices_k = tf.nn.top_k(predicted_probs, 10)
      correct_labels_tmp = tf.expand_dims(correct_labels, 1)
      correct_labels_tmp_tiled = tf.tile(correct_labels_tmp, [1, 10])
      correct_labels_reshaped = tf.reshape(correct_labels_tmp_tiled, [-1])
      predicted_labels = tf.reshape(label_indices_k, [-1])
      correct_labels = correct_labels_reshaped
      #predicted_labels = tf.Print(predicted_labels, [correct_labels, predicted_labels], summarize=100, message='Debug')
      predicted_probs = tf.reshape(predicted_topk, [-1])
      #target_weights = tf.expand_dims(target_weights, 1)
      #target_weights = tf.tile(target_weights, [1, 10])
      #target_weights = tf.reshape(target_weights, [-1])
      predicted_labels = tf.to_int64(predicted_labels)

    correct_mask = tf.where(tf.equal(correct_labels, predicted_labels), tf.ones(tf.shape(correct_labels)), tf.zeros(tf.shape(correct_labels)))
    sigma = 0.2
    print ('Corect_mask: ', correct_mask)
    def tf_kernel(matrix):
      #width = tf.Print(width, [width], message='Width of the kernel', summarize=100)
      return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  #+ tf.exp(-1.0*tf.square(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2*0.2))

    k = tf.to_int32(tf.reduce_sum(correct_mask))
    k_p = tf.to_int32(tf.reduce_sum(1.0 - correct_mask))

    print (' K : ', k)
    print (' K_p: ', k_p)

    cond_k = tf.where(tf.equal(k, 0), 0, 1)
    cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)

    #k = tf.Print(k, [k, k_p], message='k k_p', summarize=100)

    k = tf.maximum(k, 1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
    k_p = tf.maximum(k_p, 1)*cond_k_p*cond_k + (1 - cond_k_p*cond_k)*(tf.shape(correct_mask)[0] - 2)
  
    correct_probs_masked = tf.where(tf.equal(correct_mask, tf.ones(tf.shape(correct_mask))), predicted_probs, -1.0*tf.ones(tf.shape(predicted_probs)))
    correct_prob, correct_index = tf.nn.top_k(correct_probs_masked, k)
    #correct_prob = tf.Print(correct_prob, [correct_prob], summarize=1000, message='Correct Prob')
    #correct_target_weights = tf.gather(target_weights, correct_index)
    
    #correct_weight_mask = tf.where(tf.equal(correct_mask, tf.ones(tf.shape(correct_mask))), target_weights, -1.0*tf.ones(tf.shape(correct_mask)))
    #correct_weight_mask = 
    #correct_prob = tf.Print(correct_prob, [correct_index], 

    #print ('Correct_ Prob: ', correct_prob, correct_target_weights)
    incorrect_prob_masked = tf.where(tf.equal(correct_mask, tf.zeros(tf.shape(correct_mask))), predicted_probs, -1.0*tf.ones(tf.shape(predicted_probs)))
    incorrect_prob, incorrect_index = tf.nn.top_k(incorrect_prob_masked, k_p)
    #incorrect_prob = tf.Print(incorrect_prob, [incorrect_prob], summarize=1000, message='Incorrect Prob')
    #incorrect_target_weights = tf.gather(target_weights, incorrect_index)
    #correct_prob = tf.Print(correct_prob, [correct_prob], summarize=10, message='correct prob')

    # quadratic_corr_prob = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.expand_dims(1.0 - correct_prob, 0))
    # quadratic_incorrect_prob = tf.matmul(tf.expand_dims(incorrect_prob, 1), tf.expand_dims(incorrect_prob, 0))

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

    #correct_width = get_median_value((correct_prob_pairs[:, :, 0] - correct_prob_pairs[:, :, 1])**2)
    #incorrect_width = get_median_value((incorrect_prob_pairs[:, :, 0] - incorrect_prob_pairs[:, :, 1])**2)
    #correct_incorrect_width = get_median_value((correct_incorrect_pairs[:, :, 0] - correct_incorrect_pairs[:, :, 1])**2)
    correct_kernel = tf_kernel(correct_prob_pairs)
    incorrect_kernel = tf_kernel(incorrect_prob_pairs)
    correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs)
  
    sampling_weights_correct = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.transpose(tf.expand_dims(1.0 - correct_prob, 1)))
    sampling_weights_incorrect = tf.matmul(tf.expand_dims(incorrect_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1)))
  
    sampling_correct_incorrect = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1)))
    correct_denom = tf.reduce_sum(1.0 - correct_prob)
    incorrect_denom = tf.reduce_sum(incorrect_prob)
    
    #correct_target_weights_temp = tf.expand_dims(correct_target_weights, 1)
    #correct_target_weights_pairs = tf.matmul(correct_target_weights_temp, tf.transpose(correct_target_weights_temp))

    #incorrect_target_weights_temp = tf.expand_dims(incorrect_target_weights, 1)
    #incorrect_target_weights_pairs = tf.matmul(incorrect_target_weights_temp, tf.transpose(incorrect_target_weights_temp))
    
    #correct_incorrect_weights_pairs = tf.matmul(correct_target_weights_temp, tf.transpose(incorrect_target_weights_temp))

    if is_topk:
      m = tf.reduce_sum(correct_mask*tf.stop_gradient(predicted_probs))
      n = tf.reduce_sum((1.0 - correct_mask)*tf.stop_gradient(predicted_probs))
    else:
      m = tf.reduce_sum(correct_mask)
      n = tf.reduce_sum(1.0 - correct_mask)
      
    total = m + n

    if is_topk:
      correct_num_occurence = tf.stop_gradient(tf.matmul(tf.expand_dims(correct_prob, 1), tf.transpose(tf.expand_dims(correct_prob,1))))
      incorrect_num_occurence = tf.stop_gradient(tf.matmul(tf.expand_dims(incorrect_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1))))
      correct_incorrect_num_occurence = tf.stop_gradient(tf.matmul(tf.expand_dims(correct_prob, 1), tf.transpose(tf.expand_dims(incorrect_prob, 1))))
    
    if not is_topk:
        mmd_error = 1.0/(m*m + 1e-5) * tf.reduce_sum(correct_kernel * sampling_weights_correct *correct_num_occurence)
        mmd_error += 1.0/(n*n + 1e-5) * tf.reduce_mean( incorrect_kernel * sampling_weights_incorrect * incorrect_num_occurence)
        mmd_error -= 2.0/(m*n + 1e-5) * tf.reduce_mean(sampling_correct_incorrect * correct_incorrect_kernel * correct_incorrect_num_occurence)
    else:
        #m = total
        #n = total
        mmd_error = 1.0/(m*m + 1e-5)  * tf.reduce_sum(correct_kernel*sampling_weights_correct * correct_num_occurence )
        mmd_error += 1.0/(n*n + 1e-5) * tf.reduce_sum(incorrect_kernel*sampling_weights_incorrect * incorrect_num_occurence)
        mmd_error -= 2.0/ (m*n + 1e-5) * tf.reduce_sum(sampling_correct_incorrect* correct_incorrect_kernel * correct_incorrect_num_occurence)
    
    #mmd_error = 1.0*mmd_error/(total*(total - 1.0))
    #return mmd_error
    #mmd_error = tf.Print(mmd_error, [tf.to_float(cond_k*cond_k_p)*mmd_error,correct_denom, incorrect_denom, correct_prob,  cond_k, predicted_labels], summarize=100, message='debug')
    #mmd_error = tf.Print(mmd_error, [mmd_error, m, n, tf.shape(correct_labels)], summarize=100, message='MMD Error Value')
    return tf.maximum(tf.stop_gradient(tf.to_float(cond_k*cond_k_p))*tf.sqrt(mmd_error + 1e-10), 0.0)
  
def calibration_mmd_loss(logits, correct_labels):
  predicted_probs = tf.nn.softmax(logits)
  range_index = tf.to_int64(tf.expand_dims(tf.range(0, tf.shape(predicted_probs)[0]), 1))
  predicted_labels = tf.argmax(predicted_probs, axis=1)

  # predicted_probs = tf.Print(predicted_probs, [predicted_probs], summarize=50, message='Predicted_probs')

  gather_index = tf.concat([range_index, tf.expand_dims(predicted_labels, 1)], axis=1)
  predicted_probs = tf.reduce_max(predicted_probs, 1)

  correct_mask = tf.where(tf.equal(correct_labels, predicted_labels), tf.ones(tf.shape(correct_labels)), tf.zeros(tf.shape(correct_labels)))
  sigma = 0.2
  print ('Corect_mask: ', correct_mask)
  def tf_kernel(matrix, width):
    width = tf.Print(width, [width], message='Width of the kernel', summarize=100)
    return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  #+ tf.exp(-1.0*tf.square(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2*0.2))

  k = tf.to_int32(tf.reduce_sum(correct_mask))
  k_p = tf.to_int32(tf.reduce_sum(1.0 - correct_mask))

  print (' K : ', k)
  print (' K_p: ', k_p)

  cond_k = tf.where(tf.equal(k, 0), 0, 1)
  cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)

  k = tf.maximum(k, 1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
  k_p = tf.maximum(k_p, 1)*cond_k_p*cond_k + (1 - cond_k_p*cond_k)*(tf.shape(correct_mask)[0] - 2)
  
  correct_prob, _ = tf.nn.top_k(predicted_probs*correct_mask, k)
  print ('Correct_ Prob: ', correct_prob)
  incorrect_prob, _ = tf.nn.top_k(predicted_probs*(1 - correct_mask), k_p)
  # correct_prob = tf.Print(correct_prob, [correct_prob], summarize=10, message='correct prob')

  # quadratic_corr_prob = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1), tf.expand_dims(1.0 - correct_prob, 0))
  # quadratic_incorrect_prob = tf.matmul(tf.expand_dims(incorrect_prob, 1), tf.expand_dims(incorrect_prob, 0))

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

  correct_width = get_median_value((correct_prob_pairs[:, :, 0] - correct_prob_pairs[:, :, 1])**2)
  incorrect_width = get_median_value((incorrect_prob_pairs[:, :, 0] - incorrect_prob_pairs[:, :, 1])**2)
  correct_incorrect_width = get_median_value((correct_incorrect_pairs[:, :, 0] - correct_incorrect_pairs[:, :, 1])**2)
  correct_kernel = tf_kernel(correct_prob_pairs, correct_width)
  incorrect_kernel = tf_kernel(incorrect_prob_pairs, incorrect_width)
  correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs, correct_incorrect_width)
  
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
  
  # mmd_error = 1.0*mmd_error/(total*(total - 1.0))
   # return mmd_error
  # mmd_error = tf.Print(mmd_error, [tf.to_float(cond_k*cond_k_p)*mmd_error,correct_denom, incorrect_denom, correct_prob,  cond_k, predicted_labels], summarize=100, message='debug')

  return tf.maximum(tf.stop_gradient(tf.to_float(cond_k*cond_k_p))*tf.sqrt(mmd_error + 1e-10), 0.0)


def calibration_unbiased_loss(logits, correct_labels):
  predicted_probs = tf.nn.softmax(logits)
  pred_labels = tf.argmax(predicted_probs, 1)
  predicted_probs = tf.reduce_max(predicted_probs, 1)
  correct_mask = tf.where(tf.equal(pred_labels, correct_labels), tf.ones(tf.shape(pred_labels)), tf.zeros(tf.shape(pred_labels)))

  c_minus_r = tf.to_float(correct_mask) - predicted_probs
  dot_product = tf.matmul(tf.expand_dims(c_minus_r, 1), tf.transpose(tf.expand_dims(c_minus_r, 1)))

  tensor1 = predicted_probs
  prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1), [1, tf.shape(tensor1)[0]]), 2)
  prob_pairs = tf.concat([prob_tiled, tf.transpose(prob_tiled, [1, 0, 2])], axis=2)

  def tf_kernel(matrix):
    #width = tf.Print(width, [width], message='Width of the kernel', summarize=100)
    return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  #+ tf.exp(-1.0*tf.square(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2*0.2))

  kernel_prob_pairs = tf_kernel(prob_pairs)
  numerator = dot_product*kernel_prob_pairs
  return tf.reduce_sum(numerator)/tf.square(tf.to_float(tf.shape(correct_mask)[0]))
  
def self_entropy(logits):
  probs = tf.nn.softmax(logits)
  log_logits = tf.log(probs + 1e-10)
  logits_log_logits = probs*log_logits
  return -tf.reduce_mean(logits_log_logits)
