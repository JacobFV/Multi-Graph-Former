import tensorflow as tf

def alive(self, tensor, threshold=0.15, c=2):
    """qualifies the tensor as alive or dead. 
    The formula makes even a one hot encoding considered alive

    state = tanh(c * ||tensor|| / √d_tensor )
    return state if state > threshold else 0.

    params:
        tensor: multidimensional tensor to qualify as dead or alive
        threshold: minimum non-zero value to indicate
        c: scaling factor. See formula above

    returns:
        rank(tensor)-1 tensor with elements values in {0}\\cup[threshold, 1)
    """
    tensor = tf.tanh( c * tf.norm(tensor, axis=-1) / tf.sqrt(tf.shape(tensor)[-1]) )
    return tf.where(tensor > threshold, tensor, tf.zeros_like(tensor))