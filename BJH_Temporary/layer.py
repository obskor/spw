import tensorflow as tf


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

def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)

def iou_coe(output, target, threshold=0.5, smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    foreground_predicted, background_predicted = tf.split(output, [1, 1], 3)
    foreground_truth, background_truth = tf.split(target, [1, 1], 3)

    axis = [1, 2, 3]
    pre = tf.cast(foreground_predicted > threshold, dtype=tf.float32)
    truth = tf.cast(foreground_truth > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou


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


def relu(name,inputs):
    active_layer = tf.nn.relu(inputs,name=name)
    return active_layer

def leaky_relu(name,inputs,alpha=0.01):
    active_layer = tf.nn.leaky_relu(inputs,alpha,name=name)
    return active_layer

def elu(name,inputs):
    active_layer = tf.nn.elu(inputs,name=name)
    return active_layer

def p_relu(name,inputs):
    alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def BatchNorm(name,inputs,training):
    BN_layer = tf.layers.batch_normalization(inputs,momentum=0.9,epsilon=0.0001,training=training,name=name)
    return BN_layer


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

def fc(name,inputs,units):
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2



