import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()
# regularizer = tf.contrib.layers.l2_regularizer(0.00001)
regularizer = None


#############################################################################################
#                                  Network Build Functions                                  #
#############################################################################################


def conv2D(name,inputs, filters, kernel_size, strides, padding='valid'):
    """
    2D Convolution
    """
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
    """
    2D Separable Convolution
    """
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
    """
    2D Transpose Convolution
    """

    W = tf.get_variable(name+'W', filter_shape, initializer=initializer,regularizer=regularizer)
    shape = tf.shape(inputs)
    batch_size = shape[0]
    output_shape2 = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
    layer = tf.nn.conv2d_transpose(inputs, filter=W, output_shape=output_shape2,strides=strides,padding=padding)

    return layer

def global_average_pooling2D(input, n_class, name):
    """
    2D Global Average Pooling
    """
    # gap_filter = resnet.create_variable('filter', shape=(1, 1, 128, 10))
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape()[-1], n_class], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer

def relu(name,inputs):
    """
    Relu Activation Function
    """
    active_layer = tf.nn.relu(inputs,name=name)
    return active_layer

def leaky_relu(name,inputs,alpha=0.01):
    """
    Leaky Relu Activation Function
    """
    active_layer = tf.nn.leaky_relu(inputs,alpha,name=name)
    return active_layer

def elu(name,inputs):
    """
    Elu Activation Function
    """
    active_layer = tf.nn.elu(inputs,name=name)
    return active_layer

def p_relu(name,inputs):
    """
    Parametric Relu Activation Function
    """
    alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg

def BatchNorm(name,inputs,training):
    """
    Batch Normalization
    """
    BN_layer = tf.layers.batch_normalization(inputs,momentum=0.9,epsilon=0.0001,training=training,name=name)
    return BN_layer

def maxpool2D(name,inputs, pool_size, strides, padding='valid'):
    """
    2D Max Pooling
    """
    MP_layer = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,name=name)
    return MP_layer

def averagepool2D(name,inputs, pool_size, strides, padding='valid'):
    """
    2D Average Pooling
    """
    AP_layer = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,name=name)
    return AP_layer

def maxout(name,inputs, num_units):
    """
    Max out
    """
    # num_units must multiple of axis
    MO_layer = tf.contrib.layers.maxout(inputs, num_units,name=name)
    return MO_layer

def concat(name,inputs,axis):
    """
    Concatenation
    """
    con_layer = tf.concat(inputs,axis,name=name)
    return con_layer

def dropout(name,inputs,drop_rate,training):
    """
    Drop Out
    """
    DP_layer = tf.layers.dropout(inputs,drop_rate,training=training,name=name)
    return DP_layer

def add(*inputs,name):
    layer = tf.add(*inputs,name=name)
    return layer

def flatten(name,inputs):
    L1 = tf.layers.flatten(inputs,name=name)
    return L1

def fc(name,inputs,units):
    """
    Fully Connected Layer
    """
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2

def mean_iou(y_pred,y_true):
    """
    Intersection Over Union for Image Segmentation Accuracy
    """
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

#############################################################################################
#                                       Loss Functions                                      #
#############################################################################################


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


def focal_loss(output, target, use_class, gamma=2, smooth=1e-8):
    """
    arXiv:1711.01506v3. Towards Automatic 3D Shape Instantiation for Deployed Stent Grafts: 2D Multiple-class and Class-imbalance Marker Segmentation with Equally-weighted Focal U-Net

    :param output: Tensor of Predicted results.
    :param target: Tensor of Ground truth.
    :param use_class: calculate loss option. If True, using all class.
    :param alpha: float. number of weight percentage
    :param gamma: focal
    :param smooth: This small value will be added to the numerator and denominator.
    :return: loss value
    """
    pixel_wise_softmax = tf.nn.softmax(output)

    foreground_predicted, background_predicted = tf.split(pixel_wise_softmax, [1, 1], 3)
    foreground_truth, background_truth = tf.split(target, [1, 1], 3)

    if use_class is True:
        fore_focal = -tf.reduce_sum((tf.ones_like(foreground_predicted)-foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted+smooth, 1e-5, 1)))
        back_focal = -tf.reduce_sum((tf.ones_like(background_predicted)-background_predicted) ** gamma * background_truth * tf.log(tf.clip_by_value(background_predicted+smooth, 1e-5, 1)))
        focal = fore_focal + back_focal
    else:
        focal = -tf.reduce_sum((tf.ones_like(foreground_predicted) - foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted + smooth, 1e-5, 1)))

    return tf.reduce_mean(focal)


def dice_loss(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    outputs = tl.act.pixel_wise_softmax(network.outputs)
    dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
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
        raise Exception("Unknown loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = 1 - tf.reduce_mean(dice)

    return dice


def cross_entropy(output, target):
    """
    cross entropy loss
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name='cross_entropy_loss'))




