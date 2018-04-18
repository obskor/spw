import tensorflow as tf
import utils
import layer


class Model:
    def __init__(self, batch_norm_mode, depth, model_root_channel=8, img_size=256, batch_size=20, n_channel=1, n_class=2):
        tf.reset_default_graph()

        self.drop_rate=tf.placeholder(tf.float32)
        self.training=tf.placeholder(tf.bool)

        self.batch_size=batch_size
        self.model_channel=model_root_channel
        self.batch_mode = batch_norm_mode
        self.depth_n = depth

        self.X=tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='X')
        self.Y=tf.placeholder(tf.float32, [None, img_size, img_size, n_class], name='Y')

        self.logits=self.neural_net()
        print(self.logits.shape)
        self.foreground_predicted, self.background_predicted=tf.split(tf.nn.softmax(self.logits), [1, 1], 3)

        self.foreground_truth, self.background_truth=tf.split(self.Y, [1, 1], 3)

        # # Cross_Entropy
        # self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        with tf.name_scope('Loss'):
            # # Dice_Loss
            self.loss=utils.dice_loss(output=self.logits, target=self.Y)
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            # # Focal_Loss
            # self.loss=utils.focal_loss(output=self.logits, target=self.Y, use_class=False, gamma=2, smooth=1e-8)

        with tf.name_scope('Metrics'):
            self.iou = utils.iou_coe(output=self.logits, target=self.Y)

        # TB
        tf.summary.scalar('loss', self.loss)

    # def neural_net(self):
    #     with tf.name_scope('down'):
    #         channel_n=self.model_channel
    #         conv1=utils.conv2D('conv1_1', self.X, channel_n, [3, 3], [1, 1], 'same')
    #         conv1=utils.BatchNorm('BN1-1', conv1, self.training)
    #         conv1=utils.p_relu('act1_1', conv1)
    #         conv1=utils.conv2D('conv1_2', conv1, channel_n, [3, 3], [1, 1], 'same')
    #         conv1=utils.BatchNorm('BN1-2', conv1, self.training)
    #         conv1=utils.p_relu('act1_2', conv1)
    #         # print(conv1.shape)
    #         pool1=utils.maxpool('pool1', conv1, [2, 2], [2, 2], 'same') # 128 x 128
    #         # print(pool1.shape)
    #
    #         channel_n *= 2  # 32
    #         conv2=utils.conv2D('conv2_1', pool1, channel_n, [3, 3], [1, 1], 'same')
    #         conv2=utils.BatchNorm('BN2-1', conv2, self.training)
    #         conv2=utils.p_relu('act2_1', conv2)
    #         conv2=utils.conv2D('conv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2=utils.BatchNorm('BN2-2', conv2, self.training)
    #         conv2=utils.p_relu('act2_2', conv2)
    #         # print(conv2.shape)
    #         pool2=utils.maxpool('pool2', conv2, [2, 2], [2, 2], 'same') # 64 x 64
    #         # print(pool2.shape)
    #
    #         channel_n *= 2  # 64
    #         conv3=utils.conv2D('conv3_1', pool2, channel_n, [3, 3], [1, 1], 'same')
    #         conv3=utils.BatchNorm('BN3-1', conv3, self.training)
    #         conv3=utils.p_relu('act3_1', conv3)
    #         conv3=utils.conv2D('conv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3=utils.BatchNorm('BN3-2', conv3, self.training)
    #         conv3=utils.p_relu('act3_2', conv3)
    #         # print(conv3.shape)
    #         pool3=utils.maxpool('pool3', conv3, [2, 2], [2, 2], 'same') # 32 x 32
    #         # print(pool3.shape)
    #
    #         channel_n *= 2  # 128
    #         conv4=utils.conv2D('conv4_1', pool3, channel_n, [3, 3], [1, 1], 'same')
    #         conv4=utils.BatchNorm('BN4-1', conv4, self.training)
    #         conv4=utils.p_relu('act4_1', conv4)
    #         conv4=utils.conv2D('conv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4=utils.BatchNorm('BN4-2', conv4, self.training)
    #         conv4=utils.p_relu('act4_2', conv4)
    #         # print(conv4.shape)
    #         pool4=utils.maxpool('pool4', conv4, [2, 2], [2, 2], 'same') # 16 x 16
    #         # print(pool4.shape)
    #
    #         channel_n *= 2  # 256
    #         conv5=utils.conv2D('conv5_1', pool4, channel_n, [3, 3], [1, 1], 'same')
    #         conv5=utils.BatchNorm('BN5-1', conv5, self.training)
    #         conv5=utils.p_relu('act5_1', conv5)
    #         conv5=utils.conv2D('conv5_2', conv5, channel_n, [3, 3], [1, 1], 'same')
    #         conv5=utils.BatchNorm('BN5-2', conv5, self.training)
    #         conv5=utils.p_relu('act5_2', conv5)
    #         # print(conv5.shape)
    #
    #     with tf.name_scope('up'):
    #         up4=utils.deconv2D('deconv4', conv5, [3, 3, channel_n // 2, channel_n], [-1, 32, 32, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #         up4=tf.reshape(up4, shape=[-1, 32, 32, channel_n // 2])
    #         # up4=utils.BatchNorm('deBN4', up4, self.training)
    #         up4=utils.p_relu('deact4', up4)
    #         # print(up4.shape)
    #         up4=utils.concat('concat4', [up4, conv4], 3)
    #         # print(up4.shape)
    #
    #         channel_n //= 2  # 128
    #         conv4=utils.conv2D('uconv4_1', up4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4=utils.BatchNorm('uBN4-1', conv4, self.training)
    #         conv4=utils.p_relu('uact4-1', conv4)
    #         conv4=utils.conv2D('uconv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4=utils.BatchNorm('uBN4-2', conv4, self.training)
    #         conv4=utils.p_relu('uact4-2', conv4)
    #         # print(conv4.shape)
    #
    #         up3=utils.deconv2D('deconv3', conv4, [3, 3, channel_n // 2, channel_n], [-1, 64, 64, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #         up3=tf.reshape(up3, shape=[-1, 64, 64, channel_n // 2])
    #         # up3=utils.BatchNorm('deBN3', up3, self.training)
    #         up3=utils.p_relu('deact3', up3)
    #         # print(up3.shape)
    #         up3=utils.concat('concat3', [up3, conv3], 3)
    #         # print(up3.shape)
    #
    #         channel_n //= 2  # 64
    #         conv3=utils.conv2D('uconv3_1', up3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3=utils.BatchNorm('uBN3-1', conv3, self.training)
    #         conv3=utils.p_relu('uact3-1', conv3)
    #         conv3=utils.conv2D('uconv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3=utils.BatchNorm('uBN3-2', conv3, self.training)
    #         conv3=utils.p_relu('uact3-2', conv3)
    #         # print(conv3.shape)
    #
    #         up2=utils.deconv2D('deconv2', conv3, [3, 3, channel_n // 2, channel_n], [-1, 128, 128, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #         up2=tf.reshape(up2, shape=[-1, 128, 128, channel_n // 2])
    #         # up2=utils.BatchNorm('deBN2', up2, self.training)
    #         up2=utils.p_relu('deact2', up2)
    #         # print(up2.shape)
    #         up2=utils.concat('concat2', [up2, conv2], 3)
    #         # print(up2.shape)
    #
    #         channel_n //= 2  # 32
    #         conv2=utils.conv2D('uconv2_1', up2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2=utils.BatchNorm('uBN2-1', conv2, self.training)
    #         conv2=utils.p_relu('uact2-1', conv2)
    #         conv2=utils.conv2D('uconv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2=utils.BatchNorm('uBN2-2', conv2, self.training)
    #         conv2=utils.p_relu('uact2-2', conv2)
    #         # print(conv2.shape)
    #
    #         up1=utils.deconv2D('deconv1', conv2, [3, 3, channel_n // 2, channel_n], [-1, 256, 256, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #         up1=tf.reshape(up1, shape=[-1, 256, 256, channel_n // 2])
    #         # up1=utils.BatchNorm('deBN1', up1, self.training)
    #         up1=utils.p_relu('deact1', up1)
    #         # print(up1.shape)
    #         up1=utils.concat('concat1', [up1, conv1], 3)
    #         # print(up1.shape)
    #
    #         channel_n //= 2  # 16
    #         conv1=utils.conv2D('uconv1_1', up1, 16, [3, 3], [1, 1], 'same')
    #         conv1=utils.BatchNorm('uBN1-1', conv1, self.training)
    #         conv1=utils.p_relu('uact1-1', conv1)
    #         conv1=utils.conv2D('uconv1_2', conv1, 16, [3, 3], [1, 1], 'same')
    #         conv1=utils.BatchNorm('uBN1-2', conv1, self.training)
    #         conv1=utils.p_relu('uact1-2', conv1)
    #
    #         out_seg=utils.conv2D('uconv1', conv1, 2, [1, 1], [1, 1], 'same')
    #         # out_seg=utils.BatchNorm('out_BN', out_seg, self.training)
    #         out_seg=tf.nn.relu(out_seg)
    #         print(out_seg.shape)
    #
    #     return out_seg

    def get_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    
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
            down_conv = [0] * self.depth_n
            down_pool = [0] * self.depth_n
            for i in range(self.depth_n):
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
        # * 크기 회복층
        # 1. Deconv-BN-Activation-Concat으로 Feature Size를 2배로 늘리고 축소 전 데이터와 합친다.
        # 2. 채널 수를 반으로 줄인 뒤 Convolution-BN-Activation 2회 반복
        # 3. Deconvolution으로 Feature Size를 두 배로 늘리고, 다음 단계에서는 채널 수를 반으로 줄인다.
        #
        ##############################################################################################
        with tf.name_scope('up'):
            next_input = conv_keep
            conv_shape = conv_keep.shape[1].value
            up_deconv = [0] * self.depth_n
            up_conv = [0] * self.depth_n
            for i in reversed(range(self.depth_n)):
                # deconv / concat
                conv_shape *= 2

                up_deconv[i] = layer.re_conv2D(name='reconv_' + str(i), inputs=next_input, output_shape=[-1, conv_shape, conv_shape, channel_n // 2])
                # 요거 대신 re_conv2D
                # up_deconv[i] = layer.deconv2D('deconv_' + str(i),  next_input, [3, 3, channel_n // 2, channel_n],
                #                      [self.batch_size, conv_shape, conv_shape, channel_n // 2], [1, 2, 2, 1], 'SAME')
                # up_deconv[i] = tf.reshape(up_deconv[i], shape=[self.batch_size, conv_shape, conv_shape, channel_n // 2])
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

            out_seg = layer.conv2D('outconv', next_input, 2, [1, 1], [1, 1], 'same')
            out_seg = layer.BatchNorm('outBN', out_seg, self.training)
            out_seg = tf.nn.sigmoid(out_seg)

        return out_seg



        # """
        # Without Batch Normalization
        # """
        #
        # channel_n=self.model_channel  # 8
        # X = self.X
        # down_layer = [0] * self.depth_n
        # down_layer2 = [0] * self.depth_n
        # down_weight1 = [0] * self.depth_n
        # down_weight2 = [0] * self.depth_n
        # up_layer_deconv = [0] * self.depth_n
        # up_layer_conv = [0] * self.depth_n
        # up_weight1 = [0] * self.depth_n
        # up_weight2 = [0] * self.depth_n
        #
        # # Down-sampling
        # for depth in range(self.depth_n):
        #     layer_name = 'DownLayer' + str(depth + 1)
        #     if depth + 1 == 1:
        #         before_channel = 1
        #     else:
        #         before_channel = channel_n // 2
        #
        #     with tf.name_scope(layer_name):
        #         # print('before :', before_channel, 'after :', channel_n)
        #         down_weight1[depth] = self.get_variable(layer_name+"_W1", [3, 3, before_channel, channel_n])
        #         down_layer[depth] = tf.nn.conv2d(X, down_weight1[depth], strides=[1, 1, 1, 1], padding='SAME')
        #
        #         if self.batch_mode == 'on':
        #             down_layer[depth] = utils.BatchNorm(layer_name+"_BN1", down_layer[depth], self.training)
        #             down_layer[depth] = tf.nn.relu(down_layer[depth])
        #         else:
        #             down_layer[depth] = tf.nn.relu(down_layer[depth])
        #
        #         down_weight2[depth] = self.get_variable(layer_name+"_W2", [3, 3, channel_n, channel_n])
        #         down_layer2[depth] = tf.nn.conv2d(down_layer[depth], down_weight2[depth], strides=[1, 1, 1, 1], padding='SAME')
        #
        #         if self.batch_mode == 'on':
        #             down_layer2[depth] = utils.BatchNorm(layer_name+"_BN2", down_layer2[depth], self.training)
        #             down_layer2[depth] = tf.nn.relu(down_layer2[depth])
        #         else:
        #             down_layer2[depth] = tf.nn.relu(down_layer2[depth])
        #
        #         X = tf.nn.max_pool(down_layer2[depth], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #
        #     channel_n *= 2
        #
        # # Keeping
        #
        # keep_weight1 = self.get_variable('KeepLayer_W1', [3, 3, channel_n//2, channel_n])
        # keep_layer = tf.nn.conv2d(X, keep_weight1, strides=[1, 1, 1, 1], padding='SAME')
        #
        # if self.batch_mode == 'on':
        #     keep_layer = utils.BatchNorm("KeepLayer_BN1", keep_layer, self.training)
        #     keep_layer = tf.nn.relu(keep_layer)
        # else:
        #     keep_layer = tf.nn.relu(keep_layer)
        #
        # keep_weight2 = self.get_variable('KeepLayer_W2', [3, 3, channel_n, channel_n])
        # keep_layer2 = tf.nn.conv2d(keep_layer, keep_weight2, strides=[1, 1, 1, 1], padding='SAME')
        #
        # if self.batch_mode == 'on':
        #     keep_layer2 = utils.BatchNorm("KeepLayer_BN2", keep_layer2, self.training)
        #
        # keep_layer2 = tf.nn.relu(keep_layer2)
        #
        # # Up-Sampling
        # X = keep_layer2
        # for depth in reversed(range(self.depth_n)):
        #     channel_n //= 2
        #     layer_name = 'UpLayer' + str(depth+1)
        #     with tf.name_scope(layer_name):
        #
        #         up_layer_deconv[depth] = tf.layers.conv2d_transpose(X, filters=channel_n, kernel_size=2, strides=2,padding='SAME')
        #
        #         if self.batch_mode == 'on':
        #             # up_layer_deconv[depth] = tf.reshape(up_layer_deconv[depth], shape=[-1, ])
        #             up_layer_deconv[depth] = utils.BatchNorm(layer_name+"_BN1", up_layer_deconv[depth], self.training)
        #
        #         # print('after deconv shape:', up_layer_deconv[depth].shape, ', depth:', depth)
        #         up_layer_deconv[depth] = tf.concat([up_layer_deconv[depth], down_layer[depth]], 3)
        #         # print('after concat shape:', up_layer_deconv[depth].shape, ', depth:', depth)
        #         up_weight1[depth] = self.get_variable(layer_name + "_W1", [3, 3, channel_n*2, channel_n])
        #         up_layer_conv[depth] = tf.nn.conv2d(up_layer_deconv[depth], up_weight1[depth], strides=[1, 1, 1, 1], padding='SAME')
        #
        #         if self.batch_mode == 'on':
        #             up_layer_conv[depth] = utils.BatchNorm(layer_name+"_BN1", up_layer_conv[depth], self.training)
        #
        #         up_layer_conv[depth] = tf.nn.relu(up_layer_conv[depth])
        #         up_weight2[depth] = self.get_variable(layer_name + "_W2", [3, 3, channel_n, channel_n])
        #         up_layer_conv[depth] = tf.nn.conv2d(up_layer_conv[depth], up_weight2[depth], strides=[1, 1, 1, 1], padding='SAME')
        #
        #         # if self.batch_mode == 'on':
        #         #     up_layer_conv[depth] = utils.BatchNorm(layer_name+"_BN2", up_layer_conv[depth], self.training)
        #         #     X = tf.nn.relu(up_layer_conv[depth])
        #         # else:
        #         #     X = tf.nn.relu(up_layer_conv[depth])
        #
        #         X = tf.nn.relu(up_layer_conv[depth])
        #
        # # Out layer
        #
        # with tf.name_scope('OutLayer'):
        #     out_weight=self.get_variable("OutLayer_W", [1,1,channel_n,2])
        #     out_layer=tf.nn.conv2d(X,out_weight, strides=[1,1,1,1], padding='SAME')
        #     out_layer=tf.nn.sigmoid(out_layer)
        #     Y_pred=out_layer
        #
        # return Y_pred
