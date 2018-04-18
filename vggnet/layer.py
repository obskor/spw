import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = None

def conv2D(name, inputs, filters, kernel_size, strides, padding='valid'):
    conv2D_layer = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name=name)
    return conv2D_layer

def max_pool(name, inputs, pool_size, strides, padding='valid'):
    mp_layer = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size,
                                       strides=strides, padding=padding, name=name)
    return mp_layer

def batch_normalization(inputs, training, name=None):
    bn_layer = tf.layers.batch_normalization(inputs, momentum=0.9, epsilon=0.0001, training=training, name=name)
    return bn_layer

def p_relu(name, inputs):
    active_layer = tf.nn.relu(inputs, name=name)
    return active_layer

def leaky_relu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def relu(inputs, name=None):
    active_layer = tf.nn.relu(inputs, name=name)
    return active_layer