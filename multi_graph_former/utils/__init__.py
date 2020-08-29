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

def string2graph(self,
                 string,
                 forward_indicator=tf.constant([1., 0.]),
                 backward_indicator=tf.constant([0., 1.])):
    """converts a string to a directed graph with forward and backward 
    relationships defined adjacent to a main connecting diagonal.

    params:
        string: a tensor: [SAMPLE, string index, value]
            eg: [[[32], [34], [264], [235]]] representing a scentence of words

    returns:
        tuple of tensors (verts, edges) representing string as directed graph
            verts: [..., string_index, value]
            edges: [..., string_index, string_index, 2]
    """

    tokens_batch_shape = tf.shape(string)[:-2]
    num_tokens = tf.shape(string)[-2]
    
    diag = tf.eye(
        num_rows=num_tokens+1,
        num_columns=num_tokens+1,
        batch_shape=tokens_batch_shape
    )
    forward_edges = diag[..., 1:, :-1, :]
    forward_edges = tf.einsum('...ij,k->...ijk', forward_edges, forward_indicator)
    backward_edges = diag[..., :-1, 1:, :]
    backward_edges = tf.einsum('...ij,k->...ijk', backward_edges, forward_indicator)

    edges = tf.constant(forward_edges + backward_edges)

    return string, edges