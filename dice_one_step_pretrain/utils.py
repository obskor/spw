import tensorflow as tf
import cv2
import numpy as np


#############################################################################################################################
#                                                    Layer Functions                                                        #
#############################################################################################################################

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = None # tf.contrib.layers.l2_regularizer(0.00001)


def conv2D(name,inputs, filters, kernel_size, strides, padding='valid'):
    conv2D = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name=name)
    return conv2D


def s_conv2D(name,inputs,filters,kernel_size,strides,padding='valid'):
    s_conv2D = tf.layers.separable_conv2d(inputs=inputs,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          use_bias=True,
                                          depthwise_initializer=initializer,
                                          depthwise_regularizer=regularizer,
                                          pointwise_initializer=initializer,
                                          pointwise_regularizer=regularizer,
                                          name=name)
    return s_conv2D


def deconv2D(name, inputs, filter_shape, output_shape, strides, padding='valid'):
    W = tf.get_variable(name+'W', filter_shape, initializer=initializer,regularizer=regularizer)
    shape = tf.shape(inputs)
    batch_size = shape[0]
    output_shape2 = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
    layer = tf.nn.conv2d_transpose(inputs, filter=W, output_shape=output_shape2,strides=strides,padding=padding)

    return layer


def re_conv2D(name, inputs, output_shape):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """

    resize_layer = tf.image.resize_nearest_neighbor(images=inputs, size=[output_shape[1], output_shape[2]], name=name+'_resizing')
    # padding_layer = tf.pad(resize_layer)
    # conv_layer = conv2D(padding_layer)

    # conv2D('conv3_1', pool2, channel_n, [3, 3], [1, 1], 'same')
    conv_layer = conv2D(name=name+'_conv', inputs = resize_layer, filters = output_shape[3], kernel_size = [3, 3], strides = [1, 1], padding='same')

    return conv_layer


def GlobalAveragePooling2D(input, n_class, name):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    :param input: a tensor of input
    :param n_class: a number of classification class
    :return: class
    """
    # gap_filter = resnet.create_variable('filter', shape=(1, 1, 128, 10))
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape()[-1], n_class], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer


def maxpool(name,inputs, pool_size, strides, padding='valid'):
    MP_layer = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,name=name)
    return MP_layer


def averagepool(name,inputs, pool_size, strides, padding='valid'):
    AP_layer = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,name=name)
    return AP_layer


def maxout(name,inputs, num_units):
    # num_units must multiple of axis
    MO_layer = tf.contrib.layers.maxout(inputs, num_units,name=name)
    return MO_layer


def concat(name,inputs,axis):
    con_layer = tf.concat(inputs,axis,name=name)
    return con_layer


def dropout(name,inputs,drop_rate,training):
    DP_layer = tf.layers.dropout(inputs,drop_rate,training=training,name=name)
    return DP_layer


def add(*inputs,name):
    layer = tf.add(*inputs,name=name)
    return layer


def flatten(name,inputs):
    L1 = tf.layers.flatten(inputs,name=name)
    return L1


def fully_connected(name,inputs,units):
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2


#############################################################################################################################
#                                                 Activation Functions                                                      #
#############################################################################################################################


def relu(name,inputs):
    active_layer = tf.nn.relu(inputs,name=name)
    return active_layer


def leaky_relu(name, x, alpha=0.2):
    x = tf.maximum(x, alpha * x, name=name)
    return x


def elu(name,inputs):
    active_layer = tf.nn.elu(inputs,name=name)
    return active_layer


def p_relu(name,inputs):
    alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def Normalization(x, norm_type, is_train, name, G=4, esp=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
            return output

        # Batch Normalization
        elif norm_type == 'Batch':
            # output = tf.contrib.layers.batch_norm(x, center=True, scale=True, decay=0.999, is_training=is_train, updates_collections=None)
            output = tf.layers.batch_normalization(x, momentum=0.9, epsilon=0.0001, training=is_train, name='BN_'+name)
            return output

        # Group Normalization
        elif norm_type == 'Group':
            with tf.name_scope('GN_'+name):
                # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
                x = tf.transpose(x, [0, 3, 1, 2])
                N, C, H, W = x.get_shape().as_list()
                G = min(G, C)

                x = tf.reshape(x, [N, G, C // G, H, W])
                mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                x = (x - mean) / tf.sqrt(var + esp)
                # per channel gamma and beta
                gamma = tf.get_variable(name+'_GN_gamma', [C],
                                        initializer=tf.constant_initializer(1.0))
                beta = tf.get_variable(name+'_GN_beta', [C],
                                       initializer=tf.constant_initializer(0.0))
                gamma = tf.reshape(gamma, [1, C, 1, 1])
                beta = tf.reshape(beta, [1, C, 1, 1])

                output = tf.reshape(x, [N, C, H, W]) * gamma + beta
                # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
                output = tf.transpose(output, [0, 2, 3, 1])
                return output

        else:
            raise NotImplementedError


#############################################################################################################################
#                                                  Accuracy Functions                                                       #
#############################################################################################################################


def IOU_(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def iou_coe(output, target, smooth=1e-5):
    # output : self.logits
    # target : self.Y
    # return : list of Batch IoU

    pixel_wise_softmax = tf.nn.softmax(output)
    foreground_predicted, _ = tf.split(pixel_wise_softmax, [1, 1], 3)

    foreground_truth, _ = tf.split(target, [1, 1], 3)

    axis = [1, 2, 3]
    pre = tf.cast(foreground_predicted > 0.51, dtype=tf.float32)
    truth = tf.cast(foreground_truth > 0.51, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = batch_iou

    return iou


#############################################################################################################################
#                                                    Loss Functions                                                         #
#############################################################################################################################


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def focal_loss(output, target, use_class, gamma=2, smooth=1e-8):
    pixel_wise_softmax = tf.nn.softmax(output)

    foreground_predicted, background_predicted = tf.split(pixel_wise_softmax, [1, 1], 3)
    foreground_truth, background_truth = tf.split(target, [1, 1], 3)

    if use_class is True:
        fore_focal = -(tf.ones_like(foreground_predicted)-foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted+smooth, 1e-5, 1))
        back_focal = -(tf.ones_like(background_predicted)-background_predicted) ** gamma * background_truth * tf.log(tf.clip_by_value(background_predicted+smooth, 1e-5, 1))
        focal = tf.reduce_mean(fore_focal + back_focal)
    else:
        focal = tf.reduce_mean(-(tf.ones_like(foreground_predicted) - foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted + smooth, 1e-5, 1)))

    return focal


def dice_loss(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    pixel_wise_softmax = tf.nn.softmax(output)

    foreground_predicted, background_predicted = tf.split(pixel_wise_softmax, [1, 1], 3)
    foreground_truth, background_truth = tf.split(target, [1, 1], 3)

    inse = tf.reduce_sum(foreground_predicted * foreground_truth, axis=axis)

    if loss_type == 'jaccard':
        l = tf.reduce_sum(foreground_predicted * foreground_predicted, axis=axis)
        r = tf.reduce_sum(foreground_truth * foreground_truth, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(foreground_predicted, axis=axis)
        r = tf.reduce_sum(foreground_truth, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)

    return 1-dice


def cross_entropy(output, target):
    """
    cross entropy loss
    :param output:
    :param target:
    :return:
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name='cross_entropy_loss'))



