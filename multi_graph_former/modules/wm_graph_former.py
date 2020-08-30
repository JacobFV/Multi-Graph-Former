from ..layers import Graph_Multihead_Attention
from ..layers import Edge_Update
from ..layers import Smart_Update

from ..utils import alive

import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class WM_Graph_Former(tfk.Model):
    """Working Memory Graph Former: 
    a neural network of three connected graphs:
    an input graph (IG), a working memory graph (WMG), and an output graph (OG).

    The IG and OG have fixed internal edge structures.
    They are not initially connected anywhere, but individual `Directed_Edge_Update`
    units can form edges between the IG and WMG and between the WMG and OG.
    IG and OG do not directly connect.

    The WMG is initialized with one vert and no edges.
    For a user specified amount of hidden updates T_stop_enc, the WMG gathers
    information from the IG. Adjacency matrices are updated both from IG to WMG
    and within the WMG. Both graphs' verts perform intragraph attention, while
    WMG verts also attend to IG verts. After updates > T_stop_enc, WMG verts and
    edges are fixed.
    
    After another user specified amount of hidden updates T_start_dec (possibly
    before T_stop_enc), OG verts begin updating with self and WMG attention. 
    Directed edges begin to be formed from WMG verts to OG verts.

    After T_finish hidden state updates, the output graph verts are returned.
    """

    def __init__(self,
                 max_WM_verts,
                 d_WM_verts=4,
                 d_WM_edges=4,
                 d_I2WM_edges=-1,
                 d_WM2O_edges=-1,
                 WM_vert_penalty=0.1,
                 WM_edge_penalty=0.02,
                 I2WM_edge_penalty=0.03,
                 WM2O_edge_penalty=0.03,
                 num_WM_root_verts=0,
                 num_heads=8,
                 d_key=8,
                 d_val=16,
                 **kwargs):
        super(WM_Graph_Former, self).__init__(**kwargs)

        if d_I2WM_edges == -1:
            d_I2WM_edges = d_WM_edges
        if d_WM2O_edges == -1:
            d_WM2O_edges = d_WM_edges

        self.max_WM_verts = max_WM_verts
        self.d_WM_verts = d_WM_verts
        self.d_WM_edges = d_WM_edges
        self.d_I2WM_edges = d_I2WM_edges
        self.d_WM2O_edges = d_WM2O_edges
        self.WM_vert_penalty = WM_vert_penalty
        self.WM_edge_penalty = WM_edge_penalty
        self.I2WM_edge_penalty = I2WM_edge_penalty
        self.WM2O_edge_penalty = WM2O_edge_penalty
        self.num_WM_root_verts = num_WM_root_verts
        self.num_heads = num_heads
        self.d_key = d_key
        self.d_val = d_val

        # for all smart update layers
        self.update_bits = 4

        self._built = False

    def build(self, input_shape=None):
        IG_shape, OG_shape = input_shape
        I_verts_shape, I_edges_shape = IG_shape
        O_verts_shape, O_edges_shape = OG_shape

        # encoding
        # IG self attention
        self.I_verts_LN_layer = tfkl.LayerNormalization()
        self.I_self_MHA_layer = Graph_Multihead_Attention(
            num_heads=self.num_heads, d_key=self.d_key, d_val=self.d_val)
        self.I_self_dense_layer = tfkl.Dense(I_verts_shape[-1] + self.update_bits)
        self.I_self_update_layer = Smart_Update()

        # update edges from IG to WMG
        self.I2WM_edge_update_layer = Edge_Update()

        # WMG attends to IG
        self.WM_verts_LN_layer = tfkl.LayerNormalization()
        self.I2WM_MHA_layer = Graph_Multihead_Attention(
            num_heads=self.num_heads, d_key=self.d_key, d_val=self.d_val)
        self.I2WM_dense_layer = tfkl.Dense(self.d_WM_verts + self.update_bits)
        self.I2WM_update_layer = Smart_Update()

        # WMG self attention
        self.WM_self_MHA_layer = Graph_Multihead_Attention(
            num_heads=self.num_heads, d_key=self.d_key, d_val=self.d_val)
        self.WM_self_dense_layer = tfkl.Dense(self.d_WM_verts + self.update_bits)
        self.WM_self_update_layer = Smart_Update()

        # update WMG internal edges
        self.WM_edge_update_layer = Edge_Update()

        # decoding
        # update edges from WMG to OG
        self.WM2O_edge_update_layer = Edge_Update()

        # OG attends to WMG
        self.O_verts_LN_layer = tfkl.LayerNormalization()
        self.WM2O_MHA_layer = Graph_Multihead_Attention(
            num_heads=self.num_heads, d_key=self.d_key, d_val=self.d_val)
        self.WM2O_dense_layer = tfkl.Dense(O_verts_shape[-1] + self.update_bits)
        self.WM2O_update_layer = Smart_Update()

        # OG self attention
        self.O_self_MHA_layer = Graph_Multihead_Attention(
            num_heads=self.num_heads, d_key=self.d_key, d_val=self.d_val)
        self.O_self_dense_layer = tfkl.Dense(O_verts_shape[-1] + self.update_bits)
        self.O_self_update_layer = Smart_Update()
        
        self._built = True

    def call(self, inputs, T_stop_enc=20, T_start_dec=10, T_finish=30, training=False):
        IG, OG = inputs
        I_verts, I_edges = IG
        O_verts, O_edges = OG
        
        if not self._built:
            self.build(((I_verts.shape, I_edges.shape),
                        (O_verts.shape, O_edges.shape)))
            
        batch_size = I_verts.shape[:-2]
        
        WM_verts = tf.concat([
            tf.ones(batch_size + (1, self.d_WM_verts)), #seed the graph
            tf.zeros(batch_size + (self.max_WM_verts-1, self.d_WM_verts))
        ], axis=-2)

        I2WM_edges = tf.zeros(batch_size + 
            (I_verts.shape[-2], self.max_WM_verts, self.d_I2WM_edges))
        WM_edges = tf.zeros(batch_size + 
            (self.max_WM_verts, self.max_WM_verts, self.d_WM_edges))
        WM2O_edges = tf.zeros(batch_size + 
            (self.max_WM_verts, O_verts.shape[-2], self.d_WM2O_edges))

        I_adj = alive(I_edges)
        I2WM_adj = alive(I2WM_edges)
        WM_adj = alive(WM_edges)
        WM2O_adj = alive(WM2O_edges)
        O_adj = alive(O_edges)

        # root vert graphs
        if self.num_WM_root_verts > 0:
            raise NotImplementedError()

        for hidden_layer in tf.range(T_finish):

            # encoding
            if hidden_layer < T_stop_enc:

                # IG self attention
                I_verts_ = self.I_verts_LN_layer(I_verts)
                I_verts_ = self.I_self_MHA_layer((I_verts_, I_verts_, I_edges, I_adj))
                I_verts_ = self.I_self_dense_layer(I_verts_)
                I_verts = self.I_self_update_layer((I_verts, I_verts_[self.update_bits:], I_verts_))

                # update edges from IG to WMG
                I2WM_edges = self.I2WM_edge_update_layer((I_verts, WM_verts, I2WM_edges, I2WM_adj))
                I2WM_adj = alive(I2WM_edges)

                # WMG attends to IG
                WM_verts_ = self.WM_verts_LN_layer(WM_verts)
                WM_verts_ = self.I2WM_MHA_layer((I_verts, WM_verts_, I2WM_edges, I2WM_adj))
                WM_verts_ = self.I2WM_dense_layer(WM_verts_)
                WM_verts = self.I2WM_update_layer((WM_verts, WM_verts_[self.update_bits:], WM_verts_))

                # WMG self attention
                WM_verts_ = self.WM_verts_LN_layer(WM_verts)
                WM_verts_ = self.WM_self_MHA_layer((WM_verts_, WM_verts_, WM_edges, WM_adj))
                WM_verts_ = self.WM_self_dense_layer(WM_verts_)
                WM_verts = self.WM_self_update_layer((WM_verts, WM_verts_[self.update_bits:], WM_verts_))

                # update WMG internal edges
                WM_edges = self.WM_edge_update_layer((WM_verts, WM_verts, WM_edges, WM_adj))
                WM_adj = alive(WM_edges)

                # TODO
                # WM_vert_penalty=0.1,
                # WM_edge_penalty=0.02,
                # I2WM_edge_penalty=0.03,

            # decoding
            if hidden_layer >= T_start_dec:
                # update edges from WMG to OG
                WM2O_edges = self.WM2O_edge_update_layer((WM_verts, O_verts, WM2O_edges, WM2O_adj))
                WM2O_adj = alive(WM2O_edges)

                # OG attends to WMG
                O_verts_ = self.O_verts_LN_layer(O_verts)
                O_verts_ = self.WM2O_MHA_layer((WM_verts, O_verts_, WM2O_edges, WM2O_adj))
                O_verts_ = self.WM2O_dense_layer(O_verts_)
                O_verts = self.WM2O_update_layer((O_verts, O_verts_[self.update_bits:], O_verts_))

                # OG self attention
                O_verts_ = self.O_verts_LN_layer(O_verts)
                O_verts_ = self.O_self_MHA_layer((O_verts_, O_verts_, O_edges, O_adj))
                O_verts_ = self.O_self_dense_layer(O_verts_)
                O_verts = self.O_self_update_layer((O_verts, O_verts_[self.update_bits:], O_verts_))
        
                # TODO
                # WM2O_edge_penalty=0.03,

        return O_verts