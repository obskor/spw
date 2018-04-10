# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from defaultLayer import (conv3d, activate, pool, deconv3d)


class Unet():
    def __init__(self, n_i_channel=1, n_class=2, cost="mfc", cost_kwargs={}, **kwargs):
        self.n_i_channel = n_i_channel
        self.n_class = n_class

        self.x = tf.placeholder(tf.float32,
                                shape=[1, 128, 256, 256, n_i_channel])  # [None, None, None, None, n_i_channel]
        self.y = tf.placeholder(tf.float32, shape=[1, 128, 256, 256, n_class])
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self._construct_unet(**kwargs)

        self.predict = self._pixel_wise_softmax(logits)

        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict, 4), tf.argmax(self.y, 4)), tf.float32))

        self.dice = 2 * tf.reduce_sum(self.predict * self.y) / (tf.reduce_sum(self.predict) + tf.reduce_sum(self.y))

    def _pixel_wise_softmax(self, logits):
        exponential_logits = tf.exp(logits)
        summed_logits = tf.reduce_sum(exponential_logits, 4, keep_dims=True)
        duplicated_logits = tf.tile(summed_logits, tf.stack([1, 1, 1, 1, tf.shape(logits)[4]]))
        soft_max_with_logits = tf.div(exponential_logits, duplicated_logits)
        return soft_max_with_logits

    def _get_cost(self, logits, cost_name, cost_kwargs):
        loss = 0

        labels = self.y
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])

        # Mean Squared Error
        if cost_name == "mse":
            prediction = tf.nn.softmax(logits)
            loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(prediction, labels), [1, 2, 3, 4]))

        # Cross Entropy
        if "ce" in cost_name:
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

            # Weighted Cross Entropy
            if cost_name == "wce":
                class_weights = cost_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
                    flat_labels = tf.multiply(flat_labels, class_weights)

                    # weighted_target = tf.reduce_sum(flat_labels, axis=1)
                    # loss = tf.multiply(loss, weighted_target)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

        # Hard Dice Coefficience
        elif cost_name == "dsc":
            epscillon = 1e-5
            prediction = self._pixel_wise_softmax(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = epscillon + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2*intersection/union)

        # Focal Loss
        elif cost_name == "fc":
            prediction = self._pixel_wise_softmax(logits)
            loss = tf.reduce_sum(-(1-prediction)*(1-prediction)*100*labels*tf.log(prediction), [1, 2, 3, 4])
            loss = tf.reduce_mean(loss)

        # Modified Focal Loss, Modified OBS Korea. KBS, BJH
        elif cost_name == "mfc":
            prediction = self._pixel_wise_softmax(logits)

            prediction_1 = tf.slice(prediction, [0, 0, 0, 0, 0], [1, 128, 256, 256, 1])
            prediction_2 = tf.slice(prediction, [0, 0, 0, 0, 1], [1, 128, 256, 256, 1])
            labels_1 = tf.slice(labels, [0, 0, 0, 0, 0], [1, 128, 256, 256, 1])
            labels_2 = tf.slice(labels, [0, 0, 0, 0, 1], [1, 128, 256, 256, 1])

            loss_a = tf.reduce_sum(- prediction_2 * prediction_2 * 50 * labels_1 * tf.log(prediction_1), [1, 2, 3, 4])
            loss_b = tf.reduce_sum(- prediction_1 * prediction_1 * 50 * labels_2 * tf.log(prediction_2), [1, 2, 3, 4])
            loss = tf.reduce_mean(loss_a + loss_b)

        return loss

    def _construct_unet(self, n_root_channel=2, b_l_number=2, kernel_size=3, pool_size=2, stride_size=1, **kwargs):
        shape = tf.shape(self.x)
        input = tf.reshape(self.x, [-1, shape[1], shape[2], shape[3], self.n_i_channel])

        n_channel = n_root_channel

        conv_dict = {}

        conv = input
        conv_dict['input'] = input

        depth = kwargs.get("n_layer", 13) // 4
        for d_number in range(depth):
            dilation_rate = 2 ** d_number

            for l_number in range(b_l_number):
                downname = "down" + str(d_number * 2 + l_number + 1)

                with tf.variable_scope(downname):
                    conv_name = "conv" + str(l_number + 1)
                    conv = conv3d(conv, n_channel, kernel_size, stride_size, dilation_rate, conv_name)
                    conv = activate(conv, self.training, self.keep_prob, conv_name)
                    # conv=tf.concat([conv, conv_dict[]], axis=-1)
                    conv_dict[downname] = conv

            n_channel *= 2

            conv_name = "max" + str(d_number + 1)

            conv = pool(conv, pool_size, conv_name)
            conv = activate(conv, self.training, self.keep_prob, conv_name)
            conv_dict[conv_name] = conv

        conv_name = "middle_conv"
        conv = conv3d(conv, n_channel, kernel_size, stride_size, 2 ** depth, conv_name)
        conv_dict[conv_name] = conv

        for d_number in range(depth - 1, -1, -1):
            tmp = depth - d_number

            n_channel //= 2

            dilation_rate = 2 ** d_number

            conv_name = "tr" + str(tmp)
            conv = deconv3d(conv, n_channel * 2, n_channel, kernel_size, conv_name)
            conv = activate(conv, self.training, self.keep_prob, conv_name)

            conc_name = 'down' + str((d_number + 1) * 2)


            conv = tf.concat([conv, conv_dict[conc_name]], 4)

            n_channel += conv_dict[conc_name].get_shape().as_list()[4]

            conv_dict[conv_name] = conv

            for l_number in range(b_l_number):
                upname = "up" + str((depth - d_number - 1) * 2 + l_number + 1)

                with tf.variable_scope(upname):
                    conv_name = "conv" + str(l_number + 1)
                    conv = conv3d(conv, n_channel, kernel_size, stride_size, dilation_rate, conv_name)
                    activate(conv, self.training, self.keep_prob, conv_name)
                    # conv=tf.concat([conv, conv_dict[]], axis=-1)
                    conv_dict[upname] = conv

        n_channel += 1

        conv = tf.concat([conv, conv_dict['input']], 4)

        output = conv3d(conv, self.n_class, 1, 1, 1, "fc", use_bias=None)
        conv_dict['output'] = output

        print(conv_dict)

        return output