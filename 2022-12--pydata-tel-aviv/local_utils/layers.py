import tensorflow as tf
keras = tf.keras

class GlobalLehmerPool(keras.layers.Layer):
    def __init__(self, epsilon=1e-4):
        super(GlobalLehmerPool, self).__init__()
        self.epsilon = epsilon
        p_init = tf.random_normal_initializer()
        self.p = tf.Variable(
            initial_value=p_init(shape=(1,), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        # Rescale the array
        global_min = tf.reduce_min(inputs) - self.epsilon
        
        if global_min > 0:
            global_min = tf.constant(0.)
           
        # if global_mean < 0 -> add global_min + epsilon
        # if global_min == 0 -> add epsilon 
        # else: add 0
        inputs -= global_min
        
        # Compute the avg
        lehmer_mean = tf.reduce_sum(inputs**self.p, axis=-2
        ) / tf.reduce_sum(
            inputs**(self.p-1), axis=-2
        )
        
        print(lehmer_mean.shape)
        
        # Add the adj. term back
        lehmer_mean += global_min
            
        return lehmer_mean