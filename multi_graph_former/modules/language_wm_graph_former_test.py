from .language_wm_graph_former import Language_WM_Graph_Former

import tensorflow as tf
import tensorflow_datasets as tfds

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for en, pt in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for en, pt in train_examples), target_vocab_size=2**13)

BUFFER_SIZE = 20000
BATCH_SIZE = 16

@tf.function
def encode(pt, en):
    pt = [tokenizer_pt.vocab_size] \
        + tokenizer_pt.encode(pt.numpy()) \
        + [tokenizer_pt.vocab_size+1]

    en = [tokenizer_en.vocab_size] \
        + tokenizer_en.encode(en.numpy()) \
        + [tokenizer_en.vocab_size+1]

    return pt, en

#def tf_encode(pt, en):
#    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
#    result_pt.set_shape([None])
#    result_en.set_shape([None])
#
#    return result_pt, result_en

max_length = 40

def filter_max_length(pt, en):
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)

train_dataset = train_examples \
                    .map(encode) \
                    .filter(filter_max_length) \
                    .cache() \
                    .shuffle(BUFFER_SIZE) \
                    .padded_batch(BATCH_SIZE) \
                    .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples \
                .map(encode) \
                .filter(filter_max_length) \
                .padded_batch(BATCH_SIZE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = 