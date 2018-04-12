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
    axis = [1, 2, 3]
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
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





'''

def c_stem_layer(inputs,training):

    stem1 = conv2D(inputs, 30, [1, 5], 1, 'VALID', training)
    stem2 = conv2D(stem1, 30, [5, 1], 1, 'VALID', training)
    stem3 = maxpool(stem2, [3, 3], 2, 'valid')
    stem4 = conv2D(stem3, 50, [1, 5], 1, 'VALID', training)
    stem5 = conv2D(stem4, 50, [5, 1], 1, 'VALID', training)
    print(stem5.shape)
    return stem5


def stem_layer(inputs,training):

    L1 = conv2D(inputs,32,[3,3],2,'VALID',training)

    L2 = conv2D(L1,32,[3,3],1,'VALID',training)

    L3 = conv2D(L2,64,[3,3],1,'SAME',training)

    L4_1 = conv2D(L3,96,[3,3],2,'VALID',training)

    L4_2 = maxpool(L3,[3,3],2,'VALID')

    L4 = concat([L4_1,L4_2],3)


    L5_1_1 = conv2D(L4,64,[1,1],1,'SAME',training)

    L5_1_2 = conv2D(L5_1_1,64,[7,1],1,'SAME',training)

    L5_1_3 = conv2D(L5_1_2,64,[1,7],1,'SAME',training)

    L5_1_4 = conv2D(L5_1_3,96,[3,3],1,'VALID',training)


    L5_2_1 = conv2D(L4,64,[1,1],1,'SAME',training)

    L5_2_2 = conv2D(L5_2_1,96,[3,3],1,'VALID',training)


    L5 = concat([L5_1_4,L5_2_2],3)


    L6_1 = maxpool(L5,[2,2],2,'VALID')

    L6_2 = conv2D(L5,192,[3,3],2,'VALID',training)

    L6 = concat([L6_1,L6_2],3)

    return L6


def inception_A(inputs,filter,n,training):

    L1_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,filter,[n,n],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)

    out_layer = concat([L1_3,L2_2,L3_1,L4_2],3)

    return out_layer


def inception_B(inputs,filter,n,training):

    L1_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)

    L1_4 = conv2D(L1_3,filter,[1,n],1,'SAME',training)

    L1_5 = conv2D(L1_4,filter,[n,1],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,filter,[1,n],1,'SAME',training)

    L2_3 = conv2D(L2_2,filter,[n,1],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)


    out_layer = concat([L1_5,L2_3,L3_1,L4_2],3)

    return out_layer


def inception_C(inputs,filter,n,training):

    L1_1 = conv2D(inputs, filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)

    L1_4_1 = conv2D(L1_3,filter,[n,1],1,'SAME',training)

    L1_4_2 = conv2D(L1_3,filter,[1,n],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2_1 = conv2D(L2_1,filter,[n,1],1,'SAME',training)

    L2_2_2 = conv2D(L2_1,filter,[1,n],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)


    out_layer = concat([L1_4_1,L1_4_2,L2_2_1,L2_2_2,L3_1,L4_2],3)

    return out_layer


def Reduction_A(inputs,m,n,k,l,training):

    L1_1 = conv2D(inputs,k,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,l,[3,3],1,'SAME',training)

    L1_3 = conv2D(L1_2,m,[3,3],2,'VALID',training)

    L2_1 = conv2D(inputs,n,[3,3],2,'VALID',training)

    L3_1 = maxpool(inputs,[3,3],2,'VALID')

    out_layer = concat([L1_3,L2_1,L3_1],3)

    return out_layer


def Reduction_B(inputs,training):

    L1_1 = conv2D(inputs,256,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,256,[1,7],1,'SAME',training)

    L1_3 = conv2D(L1_2,320,[7,1],1,'SAME',training)

    L1_4 = conv2D(L1_3,320,[3,3],2,'VALID',training)


    L2_1 = conv2D(inputs,192,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,192,[3,3],2,'VALID',training)


    L3_1 = maxpool(inputs, [3,3],2,'VALID')

    out_layer = concat([L1_4,L2_2,L3_1],3)

    return out_layer


def auxiliary_classifiers(inputs, filter, drop_rate, training):

    L1 = conv2D(inputs,filter,[3,3],1,'SAME',training)

    L2 = maxpool(L1,[2,2],2,'SAME')

    L3 = conv2D(L2,filter*1.2,[3,3],1,'SAME',training)

    L4 = maxpool(L3,[2,2],2,'SAME')

    flat = flatten(L4)

    L5 = fc(flat,650)

    L6 = dropout(L5,drop_rate,training)

    L7 = fc(L6,2)

    return L7
'''