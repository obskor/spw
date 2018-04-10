# -*- coding: utf-8 -*-

"""
Training Module, Made by KBS, BJH. JYJ OBS Korea
"""

import Unet
import os
import tensorflow as tf
from Oracle_connector import OraDB
from datetime import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer():

    def __init__(self, data_loader, model_id, learning_rate, cost_name, optimizer, act_func, layer_n, opt_kwargs={}):
        self.data_loader = data_loader
        self.net = Unet.Unet(cost=cost_name, act_func=act_func, layer_n=layer_n, cost_kwargs={"class_weights": [1e-6, 1 - 1e-6]})
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_id = str(model_id)
        self.opt_kwargs = opt_kwargs
        self.model_save_path = '/home/user01/Javis_dl_system/models/I66/' + self.model_id
        if os.path.isdir(self.model_save_path) is False:
            os.makedirs(self.model_save_path)
            # print(os.path.isdir(self.model_save_path))
        self.net_path = self.model_save_path + '/' + self.model_id + '.ckpt'


    def _get_optimizer(self, global_step, n_t_iters):
        if self.optimizer == "adam":

            learning_rate = self.learning_rate
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.9)

            exponential_decay_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=n_t_iters * 3 * 4 * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            self.optimizer =  tf.train.AdamOptimizer(
                learning_rate=exponential_decay_learning_rate,
                **self.opt_kwargs
            ).minimize(
                self.net.cost,
                global_step=global_step
            )

        elif self.optimizer == "momentum":
            learning_rate = self.learning_rate
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            exponential_decay_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=n_t_iters * 3 * 4 * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=exponential_decay_learning_rate,
                momentum=momentum,
                **self.opt_kwargs
            ).minimize(
                self.net.cost,
                global_step=global_step
            )

        elif self.optimizer == "adagrad":

            learning_rate = self.learning_rate
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.9)

            exponential_decay_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=n_t_iters * 3 * 4 * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            self.optimizer =  tf.train.AdagradOptimizer(
                learning_rate=exponential_decay_learning_rate,
                **self.opt_kwargs
            ).minimize(
                self.net.cost,
                global_step=global_step
            )

        elif self.optimizer == "adadelta":

            learning_rate = self.learning_rate
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.9)

            exponential_decay_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=n_t_iters * 3 * 4 * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            self.optimizer =  tf.train.AdadeltaOptimizer(
                learning_rate=exponential_decay_learning_rate,
                **self.opt_kwargs
            ).minimize(
                self.net.cost,
                global_step=global_step
            )

    def train(self, n_epochs, n_t_iters, n_v_iters, b_size=1, keep_prob=1.0):
        # tf.reset_default_graph()
        global_step = tf.Variable(0, trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._get_optimizer(global_step, n_t_iters)

        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            ckpt_st = tf.train.get_checkpoint_state(self.net_path)

            if ckpt_st is not None:
                saver.restore(sess, ckpt_st.model_checkpoint_path)

            for epoch in range(n_epochs):

                # epoch 진행 상황 저장

                total_acc = 0
                total_loss = 0

                for up_down in range(2):

                    for crop_number in range(4):

                        offset = (crop_number + (up_down + epoch * 2) * 4) * (n_t_iters // 8 * 3 + n_v_iters // 8)

                        for t_index in range(n_t_iters // 8):

                            for augment_number in range(3):

                                step = offset + t_index * 3 + augment_number + 1
                                total_step = offset + t_index * 3 + 3 + 1
                                x, y = self.data_loader.load_batch(b_size=1, up_down=up_down, crop_number=crop_number, augment_number=augment_number, training=True, testing=False)
                                loss, _ = sess.run((self.net.cost, self.optimizer), feed_dict={self.net.x: x, self.net.y: y, self.net.training: True, self.net.keep_prob: keep_prob})

                                total_loss += loss / total_step

                                print("step {:} : loss {:.4f}".format(step, loss), 'Training')

                        for v_index in range(n_v_iters // 8):

                            step = offset + n_t_iters // 8 * 3 + v_index + 1  # 3 + crop_number
                            total_step = offset + n_t_iters // 8 * 3 + v_index + 1

                            x, y, files_list = self.data_loader.load_batch(b_size=1, up_down=up_down, crop_number=crop_number, augment_number=0, training=False, testing=True)  # testing=False
                            predictions, accuracy, dice = sess.run(
                                (self.net.predict, self.net.accuracy, self.net.dice),
                                feed_dict={
                                    self.net.x: x,
                                    self.net.y: y,
                                    self.net.training: False,
                                    self.net.keep_prob: keep_prob
                                }
                            )

                            total_acc += accuracy / total_step
                            print('validation')
                            print("step {:} : accuracy {:.4f} dice {:.4f}".format(step, accuracy, dice), 'Validation')

                # 에폭별로 학습이 종료되면 에폭의 평균 loss, 평균 accuracy를 db에 업데이트
                progress = epoch+1 // n_epochs * 100

                cur = OraDB.prepareCursor()
                cur.execute("update training set epoch_cost=:epoch_cost, epoch_accuracy=:epoch_accuracy, epoch_num=:epoch_num, tr_progress=:tr_progress, ed_runtime=:ed_runtime where tr_model_id=:tr_model_id limit 1",
                               [round(total_loss, 4), round(total_acc, 4), epoch + 1, progress, datetime.today().strftime("%Y%m%d%H%M%S")])

                print('epoch_cost', round(total_loss, 4), 'epoch_accuracy', round(total_acc, 4),
                      'epoch_num', epoch + 1, 'tr_progress', progress, 'ed_runtime', datetime.today().strftime("%Y%m%d%H%M%S"))

                OraDB.dbCommit()
                OraDB.releaseConn()

                print('db writing finished')

                saver.save(sess, self.net_path)
                print('model saved')

        tf.reset_default_graph()
        sess.close()

