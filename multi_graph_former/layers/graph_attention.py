import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class Graph_Attention(tfkl.Layer):
    """Performs relation-aware, vert-centric single head attention and update.
    Every dst vert makes a query that is differentiably matched to most similar
    key generated from src. The value associated with that key is returned for
    the dst vert. Keys and values depend on both the quieried vert and its edge.

    Verts must be connected for queries to succeed.

    dst verts can belong to a different graph than src verts and have different
    dimensionalities. To perform intragraph self-attention updates, make src verts
    equal to dst verts.

    For attention without consideration for edges or graph structure, try
    tf.keras.layers.Attention with queries generated from dst_verts and keys 
    and values from src_verts.
    """

    def __init__(self,
                 d_key=8,
                 d_val=16,
                 **kwargs):
        super(Graph_Attention, self).__init__(**kwargs)
        self.d_key = d_key
        self.d_val = d_val

        self._built = False

    def build(self, input_shape=None):
        self.query_layer = tfkl.Dense(self.d_key,
            activation=tf.nn.elu, use_bias=True)
        self.key_layer = tfkl.Dense(self.d_key,
            activation=tf.nn.elu, use_bias=False)
        self.val_layer = tfkl.Dense(self.d_val,
            activation=tf.nn.elu, use_bias=False)
                
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
            updated tensor representing dst_verts [..., dst, val]
        """
        src_verts, dst_verts, edges, adj = inputs

        if not self._built:
            src_verts_shape = tf.shape(src_verts)
            dst_verts_shape = tf.shape(dst_verts)
            edges_shape = tf.shape(edges)
            self.build((src_verts_shape, dst_verts_shape, edges_shape))

        # Multihead Attention
        # generate queries
        queries = self.query_layer(dst_verts)
        # queries: [SAMPLE, vert, query]
        # the vert indicated by axis:-2 should be interpretted as dst

        # vert-centric incoming neighbors
        vert_incoming = tf.einsum('...sd,...sv->...sdv', adj, src_verts)
        # vert_incoming: [..., src, dst, val]

        # generate keys
        keys = self.key_layer(edges)
        # keys: [SAMPLE, src, dst, key]

        # generate values
        vals = self.val_layer(tf.concat([vert_incoming, edges], axis=-1))
        # keys: [SAMPLE, src, dst, val]
        
        # compute attention weights from query-key dot-prod similarity
        att_ws = tf.einsum('...dk,...sdk->...sd', queries, keys)
        num_src_verts = tf.shape(src_verts)[-2]
        att_ws = tf.nn.softmax(att_ws / tf.sqrt(tf.cast(num_src_verts, tf.float32)))
        # att_ws: [..., src, dst]

        # apply attention to vert-centric values
        weighted_vs = tf.einsum('...,...v->...v', att_ws, vals)
        # att_vs: [..., src, dst, weighted-val]

        # pool attended values and merge heads
        att_vs = tf.reduce_sum(weighted_vs, axis=-2)
        # att_vs: [..., dst, pooled attended vals]

        return att_vs

class Graph_Multihead_Attention(tfkl.Layer):
    """Multihead convenience implimentation of Graph_Attention"""

    def __init__(self,
                 num_heads=8,
                 d_key=8,
                 d_val=16,
                 **kwargs):
        super(Graph_Multihead_Attention, self).__init__(**kwargs)
        self.d_key = d_key
        self.d_val = d_val

        self.attention_sublayers = [
            Graph_Attention(d_key=d_key, d_val=d_val)
            for _ in range(num_heads)]
        
        self._built = False

    def call(self, inputs):
        """
        params:
            inputs: tuple of tensors (src_verts, dst_verts, edges, adj)
                src_verts: tensor [..., src, val]
                dst_verts: tensor [..., dst, val]
                edges: tensor [..., src, dst, val]
                adj: tensor [..., src, dst]
        
        returns:
            updated tensor representing dst_verts [..., dst, val]
        """
        return tf.concat([
            attn_layer(inputs) for attn_layer in self.attention_sublayers
        ], axis=-1)