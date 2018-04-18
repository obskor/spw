import tensorflow as tf

class FasterRCnn:
    def __init__(self):
        self.X = tf.placeholder(dtype='float32', shape=[-1, 256, 256, 1])
        self.Y = tf.placeholder(dtype='float32', shape=[-1, 256, 256, 1])


    def neural_net(self):
        with tf.name_scope('fc_layer1'):
            pass

