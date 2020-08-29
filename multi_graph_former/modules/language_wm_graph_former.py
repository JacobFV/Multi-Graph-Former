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
                 d_emb,
                 out_vocab_size,
                 out_length
                 **kwargs):
        super(Language_WM_Graph_Former, self).__init__(**kwargs)

        self.enc = tfkl.Embedding(in_vocab_size, d_emb)
        self.dec = tfkl.Dense(out_vocab_size, tf.nn.swish)

        self.d_emb = d_emb
        self.out_length = out_length

    def call(inputs, **kwargs):
        """
        params:
            inputs: tensor [batch_shape, index].
                tokenization should be prepadded so that
                all elements are the same length.
                eg: [[1,2,3,75,0],[2,4,565,2,6]]

        returns:
            token softmax(logits) tensor [batch_shape, index, word]
        """

        batch_shape = inputs.shape[:-1]

        I_verts = self.enc(inputs)
        I_edges = seq_edges(inputs.shape[-2], batch_shape)

        O_verts = tf.zeros(batch_shape+(self.out_length, self.d_emb))
        O_edges = seq_edges(self.out_length, batch_shape)

        O_verts = super(Language_WM_Graph_Former, self).call(**kwargs)

        logits = self.dec(O_verts)
        return tf.nn.softmax(logits)