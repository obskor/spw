import tensorflow as tf
import utils
import layer


class Model:
    def __init__(self, model_root_channel=4, img_size=256, batch_size=20, n_channel=1, n_class=2):
        tf.reset_default_graph()

        self.drop_rate = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.batch_size = batch_size
        self.model_channel = model_root_channel
        self.depth = 4

        self.X = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='X')
        self.Y = tf.placeholder(tf.float32, [None, img_size, img_size, n_class], name='Y')

        self.predicted = self.neural_net()
        self.truth, _ = tf.split(self.Y, [1, 1], 3)

        self.loss = self.mean_focal_loss(logits=self.predicted, labels=self.truth)
        self.accuracy = layer.iou_coe(output=self.predicted, target=self.truth)
        # self.accuracy = utils.mean_iou(y_pred=self.predicted, y_true=self.truth)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def neural_net(self):
        ##############################################################################################
        # * 크기 축소층
        # 1. Convolution-BN-Activation 2회 반복
        # 2. Max-Pooling으로 Feature Size를 반으로 줄이고, 다음 단계에서는 채널 수를 2배로 늘린다.
        # 3. 1~2를 지정한 depth 횟수만큼 반복한다.
        ##############################################################################################
        with tf.name_scope('down'):
            channel_n = self.model_channel
            next_input = self.X
            down_conv = [0] * self.depth
            down_pool = [0] * self.depth
            for i in range(self.depth):
                down_conv[i] = layer.conv2D('conv' + str(i) + '_1', next_input, channel_n, [3, 3], [1, 1], 'same')
                down_conv[i] = layer.BatchNorm('BN' + str(i) + '_1', down_conv[i], self.training)
                down_conv[i] = layer.p_relu('act' + str(i) + '_1', down_conv[i])
                down_conv[i] = layer.conv2D('conv' + str(i) + '_2', down_conv[i], channel_n, [3, 3], [1, 1], 'same')
                down_conv[i] = layer.BatchNorm('BN' + str(i) + '_2', down_conv[i], self.training)
                down_conv[i] = layer.p_relu('act' + str(i) + '_2', down_conv[i])

                # depth 4 기준 256->128->64->32->16
                down_pool[i] = layer.maxpool('pool1', down_conv[i], [2, 2], [2, 2], 'same')

                channel_n *= 2
                next_input = down_pool[i]

        with tf.name_scope('keep'):
            conv_keep = layer.conv2D('conv_keep_1', next_input, channel_n, [3, 3], [1, 1], 'same')
            conv_keep = layer.BatchNorm('BN5_1', conv_keep, self.training)
            conv_keep = layer.p_relu('act5_1', conv_keep)
            conv_keep = layer.conv2D('conv_keep_2', conv_keep, channel_n, [3, 3], [1, 1], 'same')
            conv_keep = layer.BatchNorm('BN5_2', conv_keep, self.training)
            conv_keep = layer.p_relu('act5_2', conv_keep)
            # print(conv_keep.shape)

        ##############################################################################################
        # * 크기 축소층
        # 1. Convolution-BN-Activation 2회 반복
        # 2. Max-Pooling으로 Feature Size를 반으로 줄이고, 다음 단계에서는 채널 수를 2배로 늘린다.
        # 3. 1~2를 지정한 depth 횟수만큼 반복한다.
        ##############################################################################################
        with tf.name_scope('up'):
            next_input = conv_keep
            conv_shape = conv_keep.shape[1].value
            up_deconv = [0] * self.depth
            up_conv = [0] * self.depth
            for i in reversed(range(self.depth)):
                # deconv / concat
                conv_shape *= 2
                up_deconv[i] = layer.deconv2D('deconv_' + str(i),  next_input, [3, 3, channel_n // 2, channel_n],
                                     [self.batch_size, conv_shape, conv_shape, channel_n // 2], [1, 2, 2, 1], 'SAME')
                up_deconv[i] = tf.reshape(up_deconv[i], shape=[self.batch_size, conv_shape, conv_shape, channel_n // 2])
                up_deconv[i] = layer.BatchNorm('deBN_' + str(i), up_deconv[i], self.training)
                up_deconv[i] = layer.p_relu('deact_' + str(i), up_deconv[i])
                up_deconv[i] = layer.concat('concat_' + str(i), [up_deconv[i], down_conv[i]], 3)
                # conv
                channel_n //= 2
                up_conv[i] = layer.conv2D('uconv' + str(i) + '_1', up_deconv[i], channel_n, [3, 3], [1, 1], 'same')
                up_conv[i] = layer.BatchNorm('uBN' + str(i) + '_1', up_conv[i], self.training)
                up_conv[i] = layer.p_relu('uact' + str(i) + '_1', up_conv[i])
                up_conv[i] = layer.conv2D('uconv' + str(i) + '_2', up_conv[i], channel_n, [3, 3], [1, 1], 'same')
                up_conv[i] = layer.BatchNorm('uBN' + str(i) + '_2', up_conv[i], self.training)
                up_conv[i] = layer.p_relu('uact' + str(i) + '_2', up_conv[i])

                next_input = up_conv[i]

            out_seg = layer.conv2D('outconv', next_input, 1, [1, 1], [1, 1], 'same')
            out_seg = layer.BatchNorm('outBN', out_seg, self.training)
            out_seg = tf.nn.sigmoid(out_seg)

        return out_seg

    # TODO 나중에 util 함수로 빼기
    # Focal Loss : sum ( -1 * (1-P)^2 * Y * log(P) ) / Count of One-Hot Label Pixel
    def mean_focal_loss(self, logits, labels):
        label_pixel_cnt = self.tensor_in_val_cnt(labels, 1)

        epsilon = 1e-8
        return tf.reduce_sum(-1 * tf.square(tf.ones(logits.shape) - logits) * labels * tf.log(logits + epsilon)) / (label_pixel_cnt + epsilon)

    # tensor 안에 val에 해당하는 값이 몇 개나 있는지 반환.
    def tensor_in_val_cnt(self, tensor, val):
        elements_equal_to_value = tf.equal(tensor, val)
        as_ints = tf.cast(elements_equal_to_value, tf.float32)
        count = tf.reduce_sum(as_ints)
        return count


    # def train(self,x_data,y_data,batch_size):
    #     return self.sess.run([self.trainer, self.loss], feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0.3, self.training: True, self.batch_size:batch_size})
    #
    # def get_accuracy(self,x_data,y_data,batch_size):
    #     return self.sess.run(self.accuracy, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False, self.batch_size:batch_size})

    # def show_result(self,test_image,batch_size):
    #     return self.sess.run(self.predicted, feed_dict={self.X:test_ima ge,self.drop_rate: 0, self.training: False, self.batch_size:batch_size})
