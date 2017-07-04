import tensorflow as tf

def minimize_and_clip(optimizer, objective, var_list=None, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_value(grad, -clip_val, clip_val), var)
    return optimizer.apply_gradients(gradients)


def concate_to_image(tensor):
    if len(tensor.shape) < 4:
        return tf.transpose(tensor, perm=[0, 2, 1])[:, :, :, None]
    # tensor: (BATCH x TIME x COUNT x WIDTH)
    # [[0,0 ; 0,0], [1,1 ; 1,1]]
    tp = tf.transpose(tensor, perm=[0, 2, 1, 3])
    shape = tf.shape(tp)
    # tp: (BATCH x COUNT x TIME x WIDTH)
    cc = tf.reshape(tp, (shape[0], shape[1], shape[2]*shape[3]))
    # cc: (BATCH x COUNT x WIDTH*TIME)
    return cc[:, :, :, None]

def weight_norms():
    norm_sums = []
    for v in tf.trainable_variables():
        norm_sums += [(v.name, tf.reduce_sum(tf.square(v)))]
    return norm_sums