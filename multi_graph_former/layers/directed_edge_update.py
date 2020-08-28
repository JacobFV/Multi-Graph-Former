import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class Directed_Edge_Update(tfkl.Layer):

    def __init__(self,
                 **kwargs):
        super(Directed_Edge_Update, self).__init__(**kwargs)
        
        self._built = False

    def build(self, input_shape):
        d_src_vert, d_dst_vert, d_edge = input_shape

        self.MLP = tfk.Sequential([
            tfkl.Dense(d_src_vert + d_dst_vert + d_edge,
                tf.nn.swish, use_bias=False),
            tfkl.Dense(d_edge, tf.nn.swish, use_bias=False)
        ])

        self._built = True

    def call(self, inputs):
        """
        params:
            inputs: tuple of tensors (src_verts, dst_verts, edges, adj)
                src_verts: tensor [..., src, val]
                dst_verts: tensor [..., dst, val]
                edges: tensor [..., src, dst, val]
                adj: tensor [..., src, dst]

        returns:
            updated edges tensor [..., src, dst, val]
        """
        src_verts, dst_verts, edges, adj = inputs

        src_verts_shape = tf.shape(src_verts)
        dst_verts_shape = tf.shape(dst_verts)
        edges_shape = tf.shape(edges)
        adj_shape = tf.shape(adj)

        LEADING_DIMS = adj_shape[:-2] # eg: (BATCH_SIZE, TIMESTEPS)
        LEADING_DIMS_OFSET = LEADING_DIMS.shape # eg: (2,)
        LEADING_DIMS_AXES = list(range(LEADING_DIMS_OFSET)) #eg: (0, 1)

        num_src_verts = src_verts_shape[-2]
        num_dst_verts = dst_verts_shape[-2]

        d_src_verts = src_verts_shape[-1]
        d_dst_verts = dst_verts_shape[-1]
        d_edge = edges_shape[-1]

        assert num_src_verts == adj_shape[-2] \
            and num_dst_verts == adj_shape[-2]
        
        if not self._built:
            self.build((d_src_vert, d_dst_vert, d_edge))

        # vert-centric incoming adjacency diagonal
        vert_adj = tf.eye(num_dst_verts) * tf.expand_dims(tf.transpose(adj, 
            LEADING_DIMS_AXES + tuple(LEADING_DIMS_OFSET + [1,0])), axis=-1)
            # adt^T: [..., dst, src]
        assert vert_adj.shape == LEADING_DIMS + (num_dst_verts, num_src_verts, num_src_verts)
        # vert_adj: [..., dst, src, src] (last two axes form sparse diag)

        # vert-centric incoming neighbors
        vert_incoming = vert_adj @ vert
        assert vert_incoming.shape == LEADING_DIMS + (N_verts, N_verts, self.d_hidden)
        # vert_incoming: [SAMPLE, dst, src, val]

        vert_outgoing = tf.transpose(vert_incoming, 
            LEADING_DIMS_AXES + tuple(LEADING_DIMS_OFSET + [1,0,2]))
        assert vert_outgoing.shape == LEADING_DIMS + (N_verts, N_verts, self.d_hidden)
        # vert_outgoing: [SAMPLE, src, dst, val]

        return self.MLP(tf.concat([vert_incoming, vert_outgoing, edges], axis=-1))