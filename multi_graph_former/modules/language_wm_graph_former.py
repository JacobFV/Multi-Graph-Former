import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from .wm_graph_former import WM_Graph_Former
from ..utils import seq_edges

class Language_WM_Graph_Former(WM_Graph_Former):
    """inputs integer token sequences and outputs 
    """

    def __init__(self,
                 in_vocab_size,
                 out_vocab_size,
                 d_emb,
                 out_length,
                 **kwargs):
        super(Language_WM_Graph_Former, self).__init__(**kwargs)

        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.d_emb = d_emb
        self.out_length = out_length

    def build(self, input_shape=None):
        IG_shape, OG_shape = input_shape
        I_verts_shape, I_edges_shape = IG_shape
        O_verts_shape, O_edges_shape = OG_shape

        super(Language_WM_Graph_Former, self).build(input_shape)

        self.enc = tfkl.Embedding(self.in_vocab_size, self.d_emb)
        self.dec = tfkl.Dense(self.out_vocab_size, tf.nn.swish)

        self._built = True

    def call(self, inputs, **kwargs):
        """
        params:
            inputs: batch of tokenized strings tensor [batch_shape, index].
                tokenization should be prepadded so that
                all elements are the same length.
                eg: [[1,2,3,0,0],[2,4,565,2,6]]

        returns:
            token logits tensor [batch_shape, index, word]
        """

        batch_shape = inputs.shape[:-1]

        I_edges = seq_edges(inputs.shape[-2], batch_shape)

        O_verts = tf.zeros(batch_shape+(self.out_length, self.d_emb))
        O_edges = seq_edges(self.out_length, batch_shape)

        if not self._built:
            super(Language_WM_Graph_Former, self).build(
                (inputs.shape, I_edges.shape), (O_verts.shape, O_edges.shape),
                **kwargs)

        I_verts = self.enc(inputs)

        O_verts = super(Language_WM_Graph_Former, self).call(
            inputs=((I_verts, I_edges), (O_verts, O_edges)), **kwargs)

        return self.dec(O_verts)