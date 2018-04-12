import cv2
import numpy as np
import tensorflow as tf


class cal:
    def __init__(self):
        self.o = tf.placeholder(tf.float32, [1, 512, 512, 1], name='Output')
        self.t = tf.placeholder(tf.float32, [1, 512, 512, 1], name='Target')
        self.iou = self.iou_coe()

    def iou_coe(self, threshold=127, smooth=1e-5):
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
        pre = tf.cast(self.o > threshold, dtype=tf.float32)
        truth = tf.cast(self.t > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
        union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
        batch_iou = (inse + smooth) / (union + smooth)
        iou = tf.reduce_mean(batch_iou)
        return iou


imga = 'D:\\Brain_Aneurysm_dataset\\abnorm\\new_train_img_label_only\\24\\img\\y\\FILE00088.png'
imgb = imga.replace('new_train_img_label_only', 'new_train_img_label_filtered')

with tf.Session() as sess:
    img_a = cv2.imread(imga, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(imgb, cv2.IMREAD_GRAYSCALE)

    img_a = np.expand_dims(img_a, axis=3)
    img_a = np.expand_dims(img_a, axis=0)

    img_b = np.expand_dims(img_b, axis=3)
    img_b = np.expand_dims(img_b, axis=0)

    C = cal()

    fd = {C.o: img_b, C.t: img_a}
    iou = sess.run(C.iou, feed_dict=fd)
    # iou = iou_coe(img_b, img_a, threshold=127)
    # sess.run(iou)

    print(iou)