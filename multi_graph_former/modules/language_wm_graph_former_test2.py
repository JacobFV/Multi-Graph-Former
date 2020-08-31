import tensorflow as tf
from .language_wm_graph_former import Language_WM_Graph_Former

VOCAB_SIZE = 100
SEQ_LENGTH = 40

languagae_wm_graph_former = Language_WM_Graph_Former(
    in_vocab_size=VOCAB_SIZE,
    out_vocab_size=VOCAB_SIZE,
    d_emb=10,
    out_length=SEQ_LENGTH,
    max_WM_verts=10,
    d_WM_verts=10
)

inputs = tf.random.uniform((4,SEQ_LENGTH), 0, VOCAB_SIZE+2)
inputs = tf.cast(inputs, tf.int64)

#print(inputs)
outputs = languagae_wm_graph_former(inputs)
print(outputs)