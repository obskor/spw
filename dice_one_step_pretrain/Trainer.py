import tensorflow as tf
import loader
import unet_first_step_model
import time
import os
import cv2
import numpy as np


class Trainer:
    def __init__(self, training_data_path, model_path, validation_percentage,
                 initial_learning_rate, decay_step,
                 decay_rate, epoch, img_size,
                 n_class, batch_size, root_channel,
                 batch_norm_mode, depth, option_name, use_gpu):

        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

        self.training_path = training_data_path
        self.model_path = model_path
        self.val_data_cnt = validation_percentage
        self.init_learning = initial_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.batch_mode = batch_norm_mode
        self.depth = depth
        self.option_name = option_name + '-first_step'

        self.data_loader = loader.DataLoader(img_size=img_size)

        print('data Loading Started')
        dstime = time.time()
        self.img_list, self.label_list, self.data_count = self.data_loader.data_list_load(self.training_path,
                                                                                          mode='train', step=1, option_name=self.option_name)
        self.shuffled_img_list, self.shuffled_label_list = self.data_loader.data_shuffle(self.img_list, self.label_list)
        detime = time.time()
        print('data Loading Complete. Consumption Time :', detime - dstime)

        print('Dataset Split Started')
        dsstime = time.time()
        self.trainX, self.trainY, self.valX, self.valY = self.data_loader.data_split(self.shuffled_img_list,
                                                                                     self.shuffled_label_list,
                                                                                     val_size=self.val_data_cnt)

        dsetime = time.time()
        print('Train Dataset Count:', len(self.trainX), 'Validation Dataset Count:', len(self.valX))
        print('data Split Complete. Consumption Time :', dsetime - dsstime)

        self.model = unet_first_step_model.Model(batch_norm_mode=self.batch_mode, depth=self.depth, img_size=img_size,
                                                 model_root_channel=root_channel, n_channel=1, n_class=n_class, batch_size=self.batch_size)

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/' + option_name + '-first')

    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning,
                                                                     global_step=global_step,
                                                                     decay_steps=self.decay_step,
                                                                     decay_rate=self.decay_rate, staircase=True,
                                                                     name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.model.loss,
                                                                                                        global_step=global_step)

    def train(self):
        global_step = tf.Variable(0, trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        train_step = int(len(self.trainX) / self.batch_size)
        val_step = int(len(self.valX) / self.batch_size)

        with tf.Session() as sess:
            saver = tf.train.Saver()

            # TB
            self.writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print("BEGIN TRAINING")

            total_training_time = 0
            for epoch in range(self.epoch_num):

                start = time.time()

                total_cost = 0
                total_vali_iou = 0
                total_vali_acc = 0
                total_vali_unfiltered_iou = 0
                step = 0

                trainX, trainY = self.data_loader.data_shuffle(self.trainX, self.trainY)

                for idx in range(train_step):
                    batch_xs_list, batch_ys_list = self.data_loader.next_batch(trainX, trainY, idx, self.batch_size)
                    # batch_xs = self.data_loader.read_image_grey_resized(batch_xs_list)
                    # batch_ys = self.data_loader.read_label_grey_resized(batch_ys_list)
                    batch_xs, batch_ys = self.data_loader.read_data(batch_xs_list, batch_ys_list, 'train')

                    # print(batch_xs.shape, batch_ys.shape)

                    # _, cost = self.model.train(batch_xs, batch_ys, train_batch_size)

                    tr_feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys,
                                    self.model.training: True, self.model.drop_rate: 0.3}

                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    total_cost += cost
                    step += 1
                    print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num, 'step:', step, 'total_step:',
                          train_step, '  mini batch loss:', cost)

                for idx in range(val_step):
                    vali_batch_xs_list, vali_batch_ys_list = self.data_loader.next_batch(self.valX, self.valY, idx,
                                                                                         self.batch_size)
                    vali_batch_xs, vali_batch_ys = self.data_loader.read_data(vali_batch_xs_list, vali_batch_ys_list, 'validation')

                    val_feed_dict = {self.model.X: vali_batch_xs, self.model.Y: vali_batch_ys,
                                     self.model.training: False, self.model.drop_rate: 0}
                    vali_iou = sess.run(self.model.iou, feed_dict=val_feed_dict)
                    vali_iou = list(vali_iou)

                    before_filtered_length = len(vali_iou)

                    total_vali_unfiltered_iou += np.mean(vali_iou)

                    iou_list = []
                    for iou in vali_iou:
                        print(iou)
                        if iou > 0.02:
                            iou_list.append(iou)

                    after_filtered_length = len(iou_list)

                    acc = after_filtered_length/before_filtered_length

                    if len(iou_list) == 0:
                        val_mean_iou = 0
                    else:
                        val_mean_iou = np.mean(iou_list)

                    total_vali_acc += acc
                    total_vali_iou += val_mean_iou

                    if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == self.epoch_num:
                        val_img_feed_dict = {self.model.X: vali_batch_xs, self.model.training: False, self.model.drop_rate: 0}
                        predicted_result = sess.run(self.model.foreground_predicted, feed_dict=val_img_feed_dict)

                        val_img_save_path = './validation_result_imgs/' + self.option_name + '/first/' + str(epoch)
                        val_img_overlay_path = os.path.join(val_img_save_path, 'overlay')
                        val_img_y_path = os.path.join(val_img_save_path, 'y')
                        val_img_label_path = os.path.join(val_img_save_path, 'label')

                        if not os.path.exists(val_img_save_path):
                            os.makedirs(val_img_save_path)
                            os.mkdir(val_img_overlay_path)
                            os.mkdir(val_img_y_path)
                            os.mkdir(val_img_label_path)

                        val_img_overlay_fullpath = val_img_overlay_path + '/' + str(idx) + '.png'
                        val_img_y_fullpath = val_img_y_path + '/' + str(idx) + '.png'
                        val_img_label_fullpath = val_img_label_path + '/' + str(idx) + '.png'

                        for idx, label in enumerate(predicted_result):
                            test_image = vali_batch_xs[idx]
                            # test_image = np.expand_dims(test_image, axis=3)
                            test_image = np.expand_dims(test_image, axis=0)

                            # print('test_image shape :', test_image.shape)
                            # print('result_image shape :', label.shape)

                            # pred_image = label
                            _, pred_image = cv2.threshold(label, 0.4, 1.0, cv2.THRESH_BINARY)
                            pred_image_original = pred_image

                            pred_image = np.expand_dims(pred_image, axis=3)
                            pred_image = np.expand_dims(pred_image, axis=0)

                            G = np.zeros([1, 256, 256, 1])
                            B = np.zeros([1, 256, 256, 1])
                            R = pred_image
                            # print('before concatenation :', R.shape)

                            pred_image = np.concatenate((B, G, R), axis=3)
                            pred_image = np.squeeze(pred_image)

                            tR = test_image
                            tG = test_image
                            tB = test_image

                            test_image = np.concatenate((tB, tG, tR), axis=3)
                            test_image = np.squeeze(test_image)

                            test_image = test_image.astype(float)
                            pred_image = pred_image * 255

                            w = 37
                            result = cv2.addWeighted(pred_image, float(100 - w) * 0.0001, test_image, float(w) * 0.0001,
                                                     0)
                            cv2.imwrite(val_img_overlay_fullpath, result * 254)
                            cv2.imwrite(val_img_y_fullpath, pred_image_original * 254)
                            cv2.imwrite(val_img_label_fullpath, vali_batch_ys[idx][:,:,0])

                end = time.time()
                training_time = end - start
                total_training_time += training_time

                # TB
                summary = sess.run(self.merged_summary, feed_dict=val_feed_dict)
                self.writer.add_summary(summary, global_step=epoch)
                tf.summary.scalar('Filtered IoU', total_vali_iou / val_step)
                tf.summary.scalar('Unfiltered IoU', total_vali_unfiltered_iou / val_step)
                tf.summary.scalar('Accuracy', total_vali_acc / val_step)

                print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num,
                      'Loss =', '{:.4f}  '.format(total_cost / train_step),
                      'Validation IoU:{:.4f}   '.format(total_vali_iou / val_step),
                      'Validation Unfiltered IoU:{:.4f}   '.format(total_vali_unfiltered_iou / val_step),
                      'Validation Accuracy:{:.4f}   '.format(total_vali_acc / val_step),
                      'Training time: {:.2f}  '.format(training_time))

                saver.save(sess, self.model_path)
                print("MODEL SAVED")

            print("TRAINING COMPLETE")

            print("TOTAL TRAINING TIME: %.3f" % total_training_time)
