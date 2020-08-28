import tensorflow as tf

def alive(self, tensor, threshold, c=2):
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


def count_alive(verts, edges):
    """counts live verts and edges in graph. differentiable

    params:
        verts: tensor: LEADING_DIMS + (N_verts, d_vert)
        edges: tensor: LEADING_DIMS + (N_verts, N_verts, d_edge)

    returns:
        a tuple (num live verts, num live edges)
    """
    
    live_verts = alive(verts, self.VERT_MIN)
    live_verts = tf.reduce_sum(live_verts, axis=-1)
    live_edges = alive(edges, self.ADJ_MIN)
    live_edges = tf.reduce_sum(live_edges, axis=[-2,-1])

    return live_verts, live_edges