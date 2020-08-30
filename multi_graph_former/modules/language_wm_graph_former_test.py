from .language_wm_graph_former import Language_WM_Graph_Former

import tensorflow as tf
import tensorflow_datasets as tfds

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
train_examples, val_examples = train_examples.take(1000), val_examples.take(100)

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for en, pt in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for en, pt in train_examples), target_vocab_size=2**13)

BUFFER_SIZE = 20000
BATCH_SIZE = 16
#@tf.function
def encode(pt, en):
    pt = [tokenizer_pt.vocab_size] \
        + tokenizer_pt.encode(pt.numpy()) \
        + [tokenizer_pt.vocab_size+1]

    en = [tokenizer_en.vocab_size] \
        + tokenizer_en.encode(en.numpy()) \
        + [tokenizer_en.vocab_size+1]

    return en, en

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

max_length = 40

@tf.function
def filter_max_length(pt, en):
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)

train_dataset = train_examples \
                    .map(tf_encode) \
                    .filter(filter_max_length) \
                    .cache() \
                    .shuffle(BUFFER_SIZE) \
                    .padded_batch(BATCH_SIZE) \
                    .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples \
                .map(tf_encode) \
                .filter(filter_max_length) \
                .padded_batch(BATCH_SIZE)

languagae_wm_graph_former = Language_WM_Graph_Former(
    in_vocab_size=tokenizer_en.vocab_size,
    out_vocab_size=tokenizer_en.vocab_size,
    d_emb=50,
    out_length=max_length-1,
    max_WM_verts=10,
    d_WM_verts=10
)

en_ex, _ = next(iter(val_dataset))
print('output', languagae_wm_graph_former(en_ex[0:1]).numpy()[0])
#print([tokenizer_en.decode(single_logits) 
#    for single_logits 
#    in languagae_wm_graph_former(en_ex).numpy() ])

#class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#    def __init__(self, d_model, warmup_steps=4000):
#        super(CustomSchedule, self).__init__()
#        
#        self.d_model = d_model
#        self.d_model = tf.cast(self.d_model, tf.float32)
#
#        self.warmup_steps = warmup_steps
#        
#    def __call__(self, step):
#        arg1 = tf.math.rsqrt(step)
#        arg2 = step * (self.warmup_steps ** -1.5)
#        
#        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

optimizer = tf.keras.optimizers.Adadelta(0.005)

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(languagae_wm_graph_former=languagae_wm_graph_former, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# restore ckpt if available
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('latest checkpoint restored')
else:
    print('training from scratch')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(
    name='train_acc')

@tf.function(input_signature=[
    tf.TensorSpec((None, None), tf.int64),
    tf.TensorSpec((None, None), tf.int64)
])
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        logits = languagae_wm_graph_former(inputs)
        loss = loss_fn(y_true=targets, y_pred=logits)

    gradients = tape.gradient(loss, languagae_wm_graph_former.trainable_variables)
    optimizer.apply_gradients(zip(gradients, languagae_wm_graph_former.trainable_variables))

    train_loss(loss)
    train_accuracy(targets, logits)

EPOCHS = 10
import time
for epoch in tf.range(EPOCHS):
    start_time = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inputs, targets)) in enumerate(train_dataset):
        train_step(inputs, targets)

        if batch % 50 == 0:
            print(f'epoch:{epoch} batch:{batch} loss{train_loss.result()} acc{train_accuracy.result()}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'saving checkpoint at {ckpt_save_path}')

    print(f'epoch {epoch} loss:{train_loss.result()} acc:{train_accuracy.result()}')
    print(f'time taken: {time.time() - start_time} seconds')
