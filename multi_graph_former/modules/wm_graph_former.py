from ..layers import Graph_Multihead_Attention
from ..layers import Directed_Edge_Update
from ..layers import Smart_Update

from ..utils import string2graph

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
                 d_O_vert,
                 OG_edges,
                 max_WM_verts,
                 d_WM_verts=4,
                 d_WM_edges=4,
                 d_I2WM_edges=-1,
                 d_WM2O_edges=-1,
                 WM_vert_penalty=0.1,
                 WM_edge_penalty=0.02,
                 I2WM_edge_penalty=0.03,
                 WM2O_edge_penalty=0.03,
                 num_WM_root_verts=1,
                 num_heads=8,
                 d_key=8,
                 d_val=16,
                 **kwargs):
        super(WM_Graph_Former, self).__init__(**kwargs)

        self.d_O_vert = d_O_vert
        self.OG_edges = OG_edges
        self.num_O_verts = tf.shape(OG_edges)[-2]
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

        self._built = False

    def build(self, input_shape=None):



        self.IG2WMG_directed_edge_update

        # build WM root vert(s) graph
        if self.num_WM_root_verts > 0:

        self.og_verts = tf.zeros(tf.shape(ig_verts)[:-2]+(self.num_O_verts, self.d_O_vert))

    def call(self, inputs, T_stop_enc, T_start_dec, T_finish):
        ig_verts, ig_edges = inputs
        
        tf.shape(ig_verts)
        if not self._built:
            self.build(tf.shape(ig_verts), tf.shape(ig_edges))

        self.og_verts.assign(tf.zeros_like(self.og_verts))

        for hidden_layer in tf.range(T_finish):

            # encoding
            if hidden_layer < T_stop_enc:
                # IG self attention

                # update edges from IG to WMG

                # WMG attends to IG

                # WMG self attention

                # update WMG internal edges

            # decoding
            if hidden_layer >= T_start_dec:
                # update edges from WMG to OG

                # OG attends to WMG

                # OG self attention
        
        return og_verts