# -*- coding: utf-8 -*-

import tensorflow as tf

def conv3d(input, channel, kernel_size, stride_size, dilation_rate, name, use_bias=True):
    conv=tf.layers.conv3d(
        input,
        channel,
        [kernel_size, kernel_size, kernel_size],
        strides=[stride_size, stride_size, stride_size],
        padding='same',
        dilation_rate=[dilation_rate, dilation_rate, dilation_rate],
        use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0.1),
        trainable=True,
        name=name
    )
    return conv

def p_relu(x, name):
    alphas = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    return pos + neg

def activate(conv, training, keep_prob, name):
    conv = tf.layers.batch_normalization(conv, training=training, name=name+"_BN")
    conv = p_relu(conv, name+"_p_relu")
    conv = tf.layers.dropout(conv, rate=1 - keep_prob, training=training)
    return conv

def pool(input, pool_size, name):
    pooled_conv=tf.layers.max_pooling3d(
        input,
        pool_size,
        2,
        name=name
    )
    return pooled_conv

def deconv3d(input, in_channel, out_channel, kernel_size, name):
    W=tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size, out_channel, in_channel], stddev=0.1))
    shape = tf.shape(input)
    tr_conv = tf.nn.conv3d_transpose(
        input,
        W,
        (shape[0], shape[1]*2, shape[2]*2, shape[3]*2, out_channel),
        [1, 2, 2, 2, 1],
        name=name
    )
    return tr_conv
