import tensorflow as tf
from vggnet.layer import *

class FasterRCnn:
    def __init__(self):
        self.X = tf.placeholder(dtype='float32', shape=[-1, 256, 256, 1])
        self.Y = tf.placeholder(dtype='float32', shape=[-1, 2])


    def neural_net(self):
        channel = 64
        with tf.name_scope('conv_layer_1'):
            conv_1_1 = conv2D('conv_1', self.X, channel, [3, 3], [1, 1], 'same')
            bn_1_1 = batch_normalization('bn_1', )
            conv_1_2 = conv2D('conv_2', conv_1, channel, [3, 3], [1, 1], 'same')

        channel *= 2
        with tf.name_scope('conv_layer_2'):

            pass

