import tensorflow as tf

import layer
import loss

class Model:
    def __init__(self, model_root_channel=8, img_size = 256, batch_size = 20, n_channel = 1, n_class = 2):

        self.drop_rate = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.batch_size = batch_size
        self.model_channel = model_root_channel

        self.X = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='X')
        self.Y = tf.placeholder(tf.float32, [None, img_size, img_size, n_class], name='Y')

        self.logits = self.neural_net()
        self.foreground_predicted, self.background_predicted = tf.split(self.logits, [1, 1], 3)
        self.foreground_truth, self.background_truth = tf.split(self.Y, [1, 1], 3)

        # # Cross_Entropy
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        self.loss = loss.dice_loss(output=self.logits, target=self.Y)

        # self.accuracy = layer.iou_coe(output=self.logits, target=self.Y)
        with tf.name_scope('Metrics'):
            self.accuracy=layer.mean_iou(self.foreground_predicted, self.foreground_truth)

        # TB
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def neural_net(self):
        with tf.name_scope('down'):
            channel_n = self.model_channel
            conv1 = layer.conv2D('conv1_1', self.X, channel_n, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('BN1-1', conv1, self.training)
            conv1 = layer.p_relu('act1_1', conv1)
            conv1 = layer.conv2D('conv1_2', conv1, channel_n, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('BN1-2', conv1, self.training)
            conv1 = layer.p_relu('act1_2', conv1)
            # print(conv1.shape)
            pool1 = layer.maxpool('pool1', conv1, [2, 2], [2, 2], 'same') # 128 x 128
            # print(pool1.shape)

            channel_n *= 2  # 32
            conv2 = layer.conv2D('conv2_1', pool1, channel_n, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('BN2-1', conv2, self.training)
            conv2 = layer.p_relu('act2_1', conv2)
            conv2 = layer.conv2D('conv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('BN2-2', conv2, self.training)
            conv2 = layer.p_relu('act2_2', conv2)
            # print(conv2.shape)
            pool2 = layer.maxpool('pool2', conv2, [2, 2], [2, 2], 'same') # 64 x 64
            # print(pool2.shape)

            channel_n *= 2  # 64
            conv3 = layer.conv2D('conv3_1', pool2, channel_n, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('BN3-1', conv3, self.training)
            conv3 = layer.p_relu('act3_1', conv3)
            conv3 = layer.conv2D('conv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('BN3-2', conv3, self.training)
            conv3 = layer.p_relu('act3_2', conv3)
            # print(conv3.shape)
            pool3 = layer.maxpool('pool3', conv3, [2, 2], [2, 2], 'same') # 32 x 32
            # print(pool3.shape)

            channel_n *= 2  # 128
            conv4 = layer.conv2D('conv4_1', pool3, channel_n, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('BN4-1', conv4, self.training)
            conv4 = layer.p_relu('act4_1', conv4)
            conv4 = layer.conv2D('conv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('BN4-2', conv4, self.training)
            conv4 = layer.p_relu('act4_2', conv4)
            # print(conv4.shape)
            pool4 = layer.maxpool('pool4', conv4, [2, 2], [2, 2], 'same') # 16 x 16
            # print(pool4.shape)

            channel_n *= 2  # 256
            conv5 = layer.conv2D('conv5_1', pool4, channel_n, [3, 3], [1, 1], 'same')
            conv5 = layer.BatchNorm('BN5-1', conv5, self.training)
            conv5 = layer.p_relu('act5_1', conv5)
            conv5 = layer.conv2D('conv5_2', conv5, channel_n, [3, 3], [1, 1], 'same')
            conv5 = layer.BatchNorm('BN5-2', conv5, self.training)
            conv5 = layer.p_relu('act5_2', conv5)
            # print(conv5.shape)

        with tf.name_scope('up'):
            up4 = layer.deconv2D('deconv4', conv5, [3, 3, channel_n // 2, channel_n], [-1, 32, 32, channel_n // 2], [1, 2, 2, 1], 'SAME')
            up4 = tf.reshape(up4, shape=[-1, 32, 32, channel_n // 2])
            # up4 = layer.BatchNorm('deBN4', up4, self.training)
            up4 = layer.p_relu('deact4', up4)
            # print(up4.shape)
            up4 = layer.concat('concat4', [up4, conv4], 3)
            # print(up4.shape)

            channel_n //= 2  # 128
            conv4 = layer.conv2D('uconv4_1', up4, channel_n, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('uBN4-1', conv4, self.training)
            conv4 = layer.p_relu('uact4-1', conv4)
            conv4 = layer.conv2D('uconv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('uBN4-2', conv4, self.training)
            conv4 = layer.p_relu('uact4-2', conv4)
            # print(conv4.shape)

            up3 = layer.deconv2D('deconv3', conv4, [3, 3, channel_n // 2, channel_n], [-1, 64, 64, channel_n // 2], [1, 2, 2, 1], 'SAME')
            up3 = tf.reshape(up3, shape=[-1, 64, 64, channel_n // 2])
            # up3 = layer.BatchNorm('deBN3', up3, self.training)
            up3 = layer.p_relu('deact3', up3)
            # print(up3.shape)
            up3 = layer.concat('concat3', [up3, conv3], 3)
            # print(up3.shape)

            channel_n //= 2  # 64
            conv3 = layer.conv2D('uconv3_1', up3, channel_n, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('uBN3-1', conv3, self.training)
            conv3 = layer.p_relu('uact3-1', conv3)
            conv3 = layer.conv2D('uconv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('uBN3-2', conv3, self.training)
            conv3 = layer.p_relu('uact3-2', conv3)
            # print(conv3.shape)

            up2 = layer.deconv2D('deconv2', conv3, [3, 3, channel_n // 2, channel_n], [-1, 128, 128, channel_n // 2], [1, 2, 2, 1], 'SAME')
            up2 = tf.reshape(up2, shape=[-1, 128, 128, channel_n // 2])
            # up2 = layer.BatchNorm('deBN2', up2, self.training)
            up2 = layer.p_relu('deact2', up2)
            # print(up2.shape)
            up2 = layer.concat('concat2', [up2, conv2], 3)
            # print(up2.shape)

            channel_n //= 2  # 32
            conv2 = layer.conv2D('uconv2_1', up2, channel_n, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('uBN2-1', conv2, self.training)
            conv2 = layer.p_relu('uact2-1', conv2)
            conv2 = layer.conv2D('uconv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('uBN2-2', conv2, self.training)
            conv2 = layer.p_relu('uact2-2', conv2)
            # print(conv2.shape)

            up1 = layer.deconv2D('deconv1', conv2, [3, 3, channel_n // 2, channel_n], [-1, 256, 256, channel_n // 2], [1, 2, 2, 1], 'SAME')
            up1 = tf.reshape(up1, shape=[-1, 256, 256, channel_n // 2])
            # up1 = layer.BatchNorm('deBN1', up1, self.training)
            up1 = layer.p_relu('deact1', up1)
            # print(up1.shape)
            up1 = layer.concat('concat1', [up1, conv1], 3)
            # print(up1.shape)

            channel_n //= 2  # 16
            conv1 = layer.conv2D('uconv1_1', up1, 16, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('uBN1-1', conv1, self.training)
            conv1 = layer.p_relu('uact1-1', conv1)
            conv1 = layer.conv2D('uconv1_2', conv1, 16, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('uBN1-2', conv1, self.training)
            conv1 = layer.p_relu('uact1-2', conv1)

            out_seg = layer.conv2D('uconv1', conv1, 2, [1, 1], [1, 1], 'same')
            # out_seg = layer.BatchNorm('out_BN', out_seg, self.training)
            out_seg = tf.nn.relu(out_seg)
            print(out_seg.shape)

        return out_seg
