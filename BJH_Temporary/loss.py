import tensorflow as tf
import numpy as np


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
        fore_focal = -(tf.ones_like(foreground_predicted)-foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted+smooth, 1e-5, 1))
        back_focal = -(tf.ones_like(background_predicted)-background_predicted) ** gamma * background_truth * tf.log(tf.clip_by_value(background_predicted+smooth, 1e-5, 1))
        focal = tf.reduce_mean(fore_focal + back_focal)
    else:
        focal = tf.reduce_mean(-(tf.ones_like(foreground_predicted) - foreground_predicted) ** gamma * foreground_truth * tf.log(tf.clip_by_value(foreground_predicted + smooth, 1e-5, 1)))

    return focal


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
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
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

