import os
import cv2
import numpy as np
import tensorflow as tf

# 180417 BJH
class cal:
    def __init__(self, shape):
        self.shape = shape
        self.o = tf.placeholder(tf.float32, [1, self.shape, self.shape, 1], name='Output')
        self.t = tf.placeholder(tf.float32, [1, self.shape, self.shape, 1], name='Target')
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
        # iou = batch_iou
        return iou


root = 'D:\\Results\\n\\200\\label'
f_list = os.listdir(root)

a_full_path = [root + '\\' + file for file in f_list]
b_full_path = [file.replace('label', 'pred') for file in a_full_path]

# imga = 'D:\\Results\\n\\48\\label\\0_2.png'
# imgb = imga.replace('label', 'pred')

with tf.Session() as sess:
    tot_iou = []

    for idx in range(len(a_full_path)):
        imga = a_full_path[idx]
        imgb = b_full_path[idx]

        img_a = cv2.imread(imga, cv2.IMREAD_GRAYSCALE)
        _, img_a = cv2.threshold(img_a, 127, 255, cv2.THRESH_BINARY)
        # print(img_a.shape)
        img_b = cv2.imread(imgb, cv2.IMREAD_GRAYSCALE)
        _, img_b = cv2.threshold(img_b, 10, 255, cv2.THRESH_BINARY)
        # print(img_b.shape)

        img_a = np.expand_dims(img_a, axis=3)
        img_a = np.expand_dims(img_a, axis=0)
        # print(img_a.shape)

        img_b = np.expand_dims(img_b, axis=3)
        img_b = np.expand_dims(img_b, axis=0)
        # print(img_b.shape)

        C = cal(shape=256)

        fd = {C.o: img_b, C.t: img_a}
        iou = round(sess.run(C.iou, feed_dict=fd), 6)
        # iou = iou_coe(img_b, img_a, threshold=127)
        # sess.run(iou)

        print(imga, iou)
        tot_iou.append(iou)



# a = np.array([[[[0, 128]]], [[[129, 0]]], [[[0, 0]]], [[[60, 255]]]])
# print(a.shape)
#



# with tf.Session() as sess:
#     a = np.array([[[[0, 128]]], [[[129, 0]]], [[[0, 0]]], [[[60, 255]]]])
#     b = tf.split(a, [1, 1], 3)
#     c, d = sess.run(b)
#     print(c.shape)
#
#     for idx, _ in enumerate(c):
#         print(idx, c[idx], c.shape, c[idx].shape)

filtered_iou = 0
cnt = 0

for iou in tot_iou:
    if iou > 0.02:
        filtered_iou += iou
        cnt += 1

print('Accucacy:', cnt/len(tot_iou))
print('IOU when correct:', filtered_iou / cnt)