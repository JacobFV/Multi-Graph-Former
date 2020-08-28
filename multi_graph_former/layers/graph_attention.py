import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class Graph_Multihead_Attention_Layer(tfkl.Layer):
    """performs vert-centric attention over incoming neighbors
    incorperating the incoming vert and incoming edge values.

    Edges must be connected to make 
    """

    def __init__(self,
                 d_key=8,
                 d_val=16,
                 num_heads=8,
                 **kwargs):
        super(Graph_Multihead_Attention_Layer, self).__init__(**kwargs)
        self.d_key = d_key
        self.d_val = d_val
        self.num_heads = num_heads
        self._built = False

    def build(self, input_shape=None):
        
        verts_shape, edges_shape, adj_shape = input_shape

        N_verts = verts_shape[-2]
        d_vert = verts_shape[-1]
        d_edge = edges_shape[-1]
        LEADING_DIMS = verts_shape[:-2]
        LEADING_DIMS_OFSET = tf.shape(LEADING_DIMS)
        LEADING_DIMS_AXES = tf.range(LEADING_DIMS_OFSET)

        # ensure (BATCH and possibly TIMESTEPS) + (N_verts,) are equal
        assert verts_shape[:-1] == edges_shape[:-2]
        # ensure the adjacency matrix can represent the edges tensor
        assert adj_shape == edges_shape[:-1]

        assert verts.shape == LEADING_DIMS + (N_verts, self.d_hidden)
        # verts: [SAMPLE, vert, val]
        assert adj.shape == LEADING_DIMS + (N_verts, N_verts)
        # adj: [SAMPLE, src, dst]

        self.query_layer = tfk.Sequential([
            tfkl.Dense(self.num_heads * self.d_key,
                activation=tf.nn.elu, use_bias=True),
            tfkl.Dense(self.num_heads * self.d_key,
                activation=tf.nn.elu, use_bias=True)])
        
        self.key_layer = tfkl.Dense(self.num_heads * self.d_key,
                            activation=tf.nn.elu, use_bias=False)
        self.val_layer = tfkl.Dense(self.num_heads * self.d_val,
                            activation=tf.nn.elu, use_bias=False)
        
        att_vs_layer = tfkl.Dense(self.d_val * self.num_heads,
            activation=tf.nn.swish, use_bias=False)
        
        self._built = True

    def call(self, inputs, training=False):
        """Performs relation-aware, vert-centric multihead attention and update.
        Every vert only has one set of queries (1 for each attention head).
        However, keys and values are unique to a vert's incoming neighbors
        because they are a function of relation type.

        params:
            inputs: tuple of tensors (verts, edges, adjacency)
                shapes are LEADING_DIMS + ...
                    verts: ... + (N_verts, d_vert)
                    edges: ... + (N_verts, N_verts, d_edge)
                    adj: ... + (N_verts, N_verts)
                adj should be a float dtype tensor because tf.cast
                is nondifferentiable. 
        
        returns:
            updated LEADING_DIMS + (N_verts, num_heads * d_val) tensor
        """
        verts, edges, adj = inputs

        verts_shape = tf.shape(verts)
        edges_shape = tf.shape(edges)
        adj_shape = tf.shape(adj)

        if not self._built:
            self.build([verts.shape, edges.shape])

        N_verts = verts_shape[-2]
        d_vert = verts_shape[-1]
        d_edge = edges_shape[-1]
        LEADING_DIMS = verts_shape[:-2]
        LEADING_DIMS_OFSET = tf.shape(LEADING_DIMS)
        LEADING_DIMS_AXES = tf.range(LEADING_DIMS_OFSET)

        vert_adj = tf.eye(N_verts) * tf.expand_dims(tf.transpose(adj, 
            LEADING_DIMS_AXES + tuple(LEADING_DIMS_OFSET + [1,0])), axis=-1)
        # adt^T: [SAMPLE, dst, src]
        assert vert_adj.shape == LEADING_DIMS + (N_verts, N_verts, N_verts)
        # vert_adj: [SAMPLE, dst, src, src] (last two axes form sparse diag)

        # vert-centric incoming neighbors
        vert_incoming = vert_adj @ verts
        assert vert_incoming.shape == LEADING_DIMS + (N_verts, N_verts, d_vert)
        # vert_incoming: [SAMPLE, dst, src, val]

        # Multihead Attention
        # generate queries
        queries = tf.reshape(self.query_nn(verts), 
            LEADING_DIMS + (N_verts, self.num_heads, self.d_key))
        queries = tf.expand_dims(queries, axis=-2)
        assert queries.shape == LEADING_DIMS + (N_verts, 1, self.d_key)
        # queries: [SAMPLE, vert, 1, query]
        # the vert indicated by axis:-3 should be interpretted as dst

        # vert-centric incoming features
        in_feat = tf.concat([vert_incoming, tf.transpose(edges,
            LEADING_DIMS + tuple(LEADING_DIMS_OFSET + [1,0,2]))], axis=-1)
        assert in_feat.shape == LEADING_DIMS + (N_verts, N_verts, d_vert + d_edge)
        # in_feat: [SAMPLE, dst, src, feature]
        
        # generate keys
        keys = self.key_nn(in_feat)
        keys = tf.reshape(keys, 
            LEADING_DIMS + (N_verts, N_verts, self.num_heads, self.d_key))
        keys = tf.transpose(keys,
            perm=LEADING_DIMS + tuple(LEADING_DIMS_OFSET + [0,2,1,3]))
        assert keys.shape == LEADING_DIMS + (N_verts, self.num_heads, N_verts, self.d_key)
        # keys: [SAMPLE, dst, head, src, key]

        # generate values
        vals = self.val_nn(in_feat)
        vals = tf.reshape(vals,
            LEADING_DIMS + (N_verts, N_verts, self.num_heads, self.d_val))
        vals = tf.transpose(vals,
            perm=LEADING_DIMS + tuple(LEADING_DIMS_OFSET + [0,2,1,3]))
        assert vals.shape == LEADING_DIMS + (N_verts, self.num_heads, N_verts, self.d_val)
        # vals: [SAMPLE, dst, head, src, val]
        
        # compute attention weights from query-key dot-prod similarity
        att_ws = queries @ tf.transpose(keys, 
            LEADING_DIMS + tuple(LEADING_DIMS_OFSET + [0,1,3,2]))
        att_ws = tf.nn.softmax(att_ws / np.sqrt(self.d_key))
        att_ws = tf.transpose(att_ws,
            LEADING_DIMS + tuple(LEADING_DIMS_OFSET + [0,1,3,2]))
        assert att_ws.shape == LEADING_DIMS + (N_verts, self.num_heads, N_verts, 1)
        # att_ws: [SAMPLE, dst, head, src, 1]

        # apply attention to vert-centric values
        att_vs = vals * att_ws
        assert att_vs.shape == LEADING_DIMS + (N_verts, self.num_heads, N_verts, self.d_val)
        # att_vs: [SAMPLE, dst, head, src, weighted-val-vec]

        # pool attended values and merge heads
        att_vs = tf.reduce_sum(att_vs, axis=-2)
        att_vs = tf.reshape(att_vs, LEADING_DIMS + (N_verts, self.num_heads * self.d_val))
        assert att_vs.shape == LEADING_DIMS + (N_verts, self.num_heads * self.d_val)
        # att_vs: [SAMPLE, dst, pooled attended vals]

        return att_vs