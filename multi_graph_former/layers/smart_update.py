import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class Smart_Update(tfkl.Layer):

    def __init__(self,
                 preprocess_layers_dims=[],
                 **kwargs):
        super(Smart_Update, self).__init__(**kwargs)
        self.preprocess_layers_dims = preprocess_layers_dims

        self._built = False

    def build(self, input_shape):
        self.preprocessor_nn = tfk.Sequential([
            tfkl.Dense(layer_dim, activation=tf.nn.swish, use_bias=False)
            for layer_dim in self.preprocess_layers_dims])
        self.updates_weight_layer = tfkl.Dense(1, activation=tf.nn.swish, use_bias=False)
        self.erase_weight_layer = tfkl.Dense(1, activation=tf.nn.swish, use_bias=False)

        self._built = True

    def call(self, inputs):
        """Erases from and writes new values to a tensor.
        
        output = (1 - w_e(`data`)) * `origonal` + w_u(`data`) * `updates`

        Params:
            inputs: tuple of tensor-like (`origonal`, `updates`) or
                (`origonal`, `updates`, `data`). If `data` is not
                supplied, `updates` is used in place of it. 

        Returns:
            A `tensor` shaped as `origonal`, `updates`, and `data`"""
        
        if not self._built:
            self.build(input_shape=tf.shape(inputs))

        if len(inputs) == 3:
            origonal, updates, data = inputs
        elif len(inputs) == 2:
            origonal, updates = inputs
            data = updates
        else:
            raise f'`len(inputs)` != 2 or 3'

        preprocessed_data = self.preprocessor_nn(data)
        weighted_updates = updates * self.updates_weight_layer(preprocessed_data)
        partially_erased = origonal * self.erase_weight_layer(preprocessed_data)
        return partially_erased + weighted_write_val