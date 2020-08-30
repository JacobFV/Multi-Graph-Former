import tensorflow as tf
import numpy as np

def alive(tensor, threshold=0.15, c=2):
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
    tensor = tf.tanh( c * tf.norm(tensor, axis=-1) / np.sqrt(tf.shape(tensor)[-1]) )
    return tf.where(tensor > threshold, tensor, tf.zeros_like(tensor))

def seq_edges(length,
              batch_shape=(1,),
              forward_indicator=tf.constant([1., 0.]),
              backward_indicator=tf.constant([0., 1.])):
    """builds edge tensor for `length`-long doubly linked noncyclic sequences

    params:
        length: length of sequence to build edge tensor for
        batch_shape: leading dimensions to repeat returned tensor for

    returns:
        edges tensor: [..., string_index, string_index, 2]
    """

    print('diag from',length, batch_shape)
    diag = tf.eye(
        num_rows=length+1,
        num_columns=length+1,
        batch_shape=batch_shape
    )
    forward_edges = diag[..., 1:, :-1, :]
    forward_edges = tf.einsum('...ij,k->...ijk', forward_edges, forward_indicator)
    backward_edges = diag[..., :-1, 1:, :]
    backward_edges = tf.einsum('...ij,k->...ijk', backward_edges, forward_indicator)

    return tf.constant(forward_edges + backward_edges)