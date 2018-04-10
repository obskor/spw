# -*- coding: utf-8 -*-

"""
Training Module, Made by KBS, BJH. JYJ OBS Korea
"""

import data_mover
import Unet
import default_Unet
import os
import tensorflow as tf
import numpy as np
import cv2
import shutil
from datetime import datetime
from Oracle_connector import OraDB


# 민스키에 올라갈 때 주석처리 해제
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class in_Tester():
    def __init__(self, data_loader, cost_name, act_func, layer_n, ckpt_path, file_path, tr_model_id, dl_test_id,
                 studylist_id, opt_kwargs={}):
        self.data_loader = data_loader

        if tr_model_id == 1:
            self.ckpt_path = '/home/user01/Javis_dl_system/models/I66/1/Unet.cpkt'
        else:
            self.ckpt_path = ckpt_path + str(tr_model_id) + '/' + str(tr_model_id) + '.ckpt'

        self.file_path = file_path
        self.dl_test_id = dl_test_id
        self.studylist_id = studylist_id

        # model loader. model id 1 : default model
        if tr_model_id == 1:
            self.net = default_Unet.Unet(cost="mfc", cost_kwargs={"class_weights": [1e-6, 1 - 1e-6]})
        else:
            self.net = Unet.Unet(cost=cost_name, act_func=act_func, layer_n=layer_n,
                                 cost_kwargs={"class_weights": [1e-6, 1 - 1e-6]})

        self.opt_kwargs = opt_kwargs

        # default ai result
        self.status = 'N'

        # oracle db information
        self.cursor = OraDB.prepareCursor()
        self.series_uid = self.cursor.execute("select series_uid from studylist where studylist_id=:studylist_id", {'studylist_id':self.studylist_id})
        self.ruserid = self.cursor.execute("select ruserid from dl_test where dl_test_id=:dl_test_id", {'dl_test_id':self.dl_test_id})

        print('tester initialized')

    # x data O, label data O, IN
    def infer1(self, n_t_iters, b_size, keep_prob=1.0):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.ckpt_path)
            print('model restored')

            t_accuracy = 0
            positions = []
            for _ in range(n_t_iters):

                for up_down in range(2):
                    for crop_number in range(4):
                        x, y, files_list, files_list_down, he, wi = self.data_loader.test_load_batch_1(b_size, up_down, crop_number)  # testing=False

                        predictions, dice = sess.run(
                            (self.net.predict, self.net.dice),
                            feed_dict={
                                self.net.x: x,
                                self.net.y: y,
                                self.net.training: False,
                                self.net.keep_prob: keep_prob
                            }
                        )

                        if crop_number == 0:
                            x_offset, y_offset = 0, 0
                        elif crop_number == 1:
                            x_offset, y_offset = 44, 0
                        elif crop_number == 2:
                            x_offset, y_offset = 0, 44
                        else:
                            x_offset, y_offset = 44, 44

                        batch_size, nz, ny, nx, _ = np.shape(predictions)
                        for index in range(batch_size):
                            files, offset = files_list[index]

                            for z_index in range(nz):
                                if offset == 0 or 128 <= offset + z_index:
                                    if offset == 0:
                                        file_path = files[z_index]
                                        file_down_path = files_list_down[z_index]
                                    else:
                                        file_path = files[offset + z_index]
                                        file_down_path = files_list_down[offset + z_index]

                                    # merge_path = file_path.replace('.jpg', '_prediction.jpg')
                                    image = cv2.imread(file_down_path)
                                    # print(file_down_path)

                                    if image is None:
                                        image = np.zeros([300, 300, 3])

                                    roi = image[y_offset: y_offset + 256, x_offset: x_offset + 256]
                                    b_prediction = predictions[index, z_index, ..., 0] > 0.5  # 255
                                    b_image = cv2.bitwise_and(roi, roi, mask=(~b_prediction * 255).astype(
                                        np.uint8))  # np.expand_dims
                                    f_image = np.stack([np.zeros([ny, nx]), np.zeros([ny, nx]), b_prediction * 255],
                                                       axis=-1)  # np.tile np.append
                                    merge = b_image + f_image
                                    # merge = np.hstack([image, label, prediction])
                                    image[y_offset: y_offset + 256, x_offset: x_offset + 256] = merge  # f_image
                                    cv2.imwrite(file_down_path, image)

                                    if crop_number == 3:
                                        position_t = cv2.imread(file_down_path)
                                        position = position_t[:, :, 2]
                                        position = cv2.resize(position, (wi, he), interpolation=cv2.INTER_AREA)

                                        _, position = cv2.threshold(position, 127, 255, cv2.THRESH_BINARY)

                                        num_labels, markers, state, cent = cv2.connectedComponentsWithStats(position)

                                        if num_labels != 1:
                                            self.status = 'Y'
                                            for idx in range(1, num_labels):
                                                x, y, w, h, size = state[idx]
                                                infor_position = [z_index, w, h, x, y]
                                                positions.append(infor_position)
                                        roi = cv2.imread(file_path)

                                        b_image = cv2.bitwise_and(roi, roi, mask=~position.astype(np.uint8))
                                        f_image = np.stack([np.zeros([he, wi]), np.zeros([he, wi]), position], axis=-1)
                                        merge = b_image + f_image
                                        cv2.imwrite(file_down_path, merge)
                        t_accuracy += dice
                print(positions)

                # ai result save
                self.cursor.execute("update dl_test set ai_result=:ai_result where dl_test_id = :dl_test_id", {'ai_result': self.status, 'dl_test_id': self.dl_test_id})
                OraDB.dbCommit()
                print(self.status)

                # test_labelinfo에 레이블 정보 저장
            for imgs in positions:
                print(imgs)
                imgindex, width, height, x_left_up, y_left_up = imgs
                x_right_down = x_left_up + width
                y_right_down = y_left_up + height

                handles = '{"start":{"x":' + str(x_left_up) + ',"y":' + str(y_left_up) + ',"highlight":true,"active":true},"end":{"x":' \
                                           + str(x_right_down) + ',"y":' + str(y_right_down) + ',"highlight":true,"active": false},"textBox":' \
                                                                                               '{"active":false,"hasMoved":false,"movesIndependently":false,' \
                                                                                               '"allowedOutsideImage": true,"hasBoundingBox": true,"x":' \
                                           + str(x_right_down) + ',"y":'+ str(y_right_down) + ',"boundingBox": {"width":' \
                                           + str(width) + ', "height": ' + str(height) + ',"left":0,"top":0}}}'


                # self.test_labelinfo_collection.insert(data)
                self.cursor.execute("insert into test_labelinfo(test_lb_id, dl_test_id, studylist_id, file_name, label_info, image_index_number, series_uid, tooltype, del_yn, createdate, ruserid) "
                                    "values(test_labelinfo_test_lb_id_seq.nextval, :dl_test_id, :studylist_id, :file_name, :label_info, :image_index_number, :series_uid, :tooltype, :del_yn, :createdate, :ruserid)"
                                    , [int(self.dl_test_id), self.studylist_id, None, handles, imgindex, self.series_uid, 'rectangleRoi', 'N', datetime.today().strftime("%Y%m%d%H%M%S"), self.ruserid])
                OraDB.dbCommit()

                print('data restored to test_labelinfo_collection')

            # dice score 저장
            a_accuracy = t_accuracy / n_t_iters / 8
            self.cursor.execute("update dl_test set dice_score=:a_accuracy where dl_test_id=:dl_test_id", [a_accuracy, self.dl_test_id])
            OraDB.dbCommit()

            print("test accuracy {:}".format(a_accuracy))

            OraDB.releaseConn()

            del_path = self.file_path + 'DownLoad'
            shutil.rmtree(del_path, ignore_errors=True)

    # x data O, label data X, IN
    def infer2(self, n_t_iters, b_size, keep_prob=1.0):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.ckpt_path)
            print('model restored')
            positions = []
            for i in range(n_t_iters):

                for up_down in range(2):
                    for crop_number in range(4):
                        x, files_list, files_list_down, he, wi = self.data_loader.test_load_batch_2(b_size, up_down, crop_number)  # testing=False

                        predictions = sess.run(self.net.predict, feed_dict={self.net.x: x, self.net.training: False, self.net.keep_prob: keep_prob})

                        if crop_number == 0:
                            x_offset, y_offset = 0, 0
                        elif crop_number == 1:
                            x_offset, y_offset = 44, 0
                        elif crop_number == 2:
                            x_offset, y_offset = 0, 44
                        else:
                            x_offset, y_offset = 44, 44

                        batch_size, nz, ny, nx, _ = np.shape(predictions)
                        for index in range(batch_size):
                            files, offset = files_list[index]

                            for z_index in range(nz):
                                if offset == 0 or 128 <= offset + z_index:
                                    if offset == 0:
                                        file_path = files[z_index]
                                        file_down_path = files_list_down[z_index]
                                    else:
                                        file_path = files[offset + z_index]
                                        file_down_path = files_list_down[offset + z_index]

                                    # merge_path = file_path.replace('.jpg', '_prediction.jpg')
                                    image = cv2.imread(file_down_path)
                                    # print(file_down_path)

                                    if image is None:
                                        image = np.zeros([300, 300, 3])

                                    roi = image[y_offset: y_offset + 256, x_offset: x_offset + 256]
                                    b_prediction = predictions[index, z_index, ..., 0] > 0.5  # 255
                                    b_image = cv2.bitwise_and(roi, roi, mask=(~b_prediction * 255).astype(
                                        np.uint8))  # np.expand_dims
                                    f_image = np.stack([np.zeros([ny, nx]), np.zeros([ny, nx]), b_prediction * 255],
                                                       axis=-1)  # np.tile np.append
                                    merge = b_image + f_image
                                    # merge = np.hstack([image, label, prediction])
                                    image[y_offset: y_offset + 256, x_offset: x_offset + 256] = merge  # f_image
                                    cv2.imwrite(file_down_path, image)

                                    if crop_number == 3:
                                        # print('file down path : ', file_down_path)
                                        # print(os.path.isfile(file_down_path))
                                        position_t = cv2.imread(file_down_path)

                                        position = position_t[:, :, 2]
                                        # print(wi, he)
                                        position = cv2.resize(position, (wi, he), interpolation=cv2.INTER_AREA)
                                        # print('position t shape : ', position_t.shape)
                                        _, position = cv2.threshold(position, 127, 255, cv2.THRESH_BINARY)

                                        num_labels, markers, state, cent = cv2.connectedComponentsWithStats(position)

                                        if num_labels != 1:
                                            self.status = 'Y'
                                            for idx in range(1, num_labels):
                                                x, y, w, h, size = state[idx]
                                                infor_position = [z_index, w, h, x, y]
                                                positions.append(infor_position)
                                        roi = cv2.imread(file_path)
                                        # print('file path : ', file_path)
                                        # print(os.path.isfile(file_path))
                                        # print('roi shape', roi.shape)
                                        b_image = cv2.bitwise_and(roi, roi, mask=~position.astype(np.uint8))
                                        f_image = np.stack([np.zeros([he, wi]), np.zeros([he, wi]), position], axis=-1)
                                        merge = b_image + f_image
                                        cv2.imwrite(file_down_path, merge)

                print(positions)


                self.cursor.execute("update dl_test set ai_result=:RES where dl_test_id = :DL_TEST_ID", [self.status, self.dl_test_id])
                OraDB.dbCommit()

                # test_labelinfo에 레이블 정보 저장
                # positions = [ [imageindex, width, height, x_start, y_start] ]
            for imgs in positions:
                print(imgs)
                imgindex, width, height, x_left_up, y_left_up = imgs

                x_right_down = x_left_up + width
                y_right_down = y_left_up + height

                handles = '{"start":{"x":' + str(x_left_up) + ',"y":' + str(y_left_up) + ',"highlight":true,"active":true},"end":{"x":' \
                                           + str(x_right_down) + ',"y":' + str(y_right_down) + ',"highlight":true,"active": false},"textBox":' \
                                                                                               '{"active":false,"hasMoved":false,"movesIndependently":false,' \
                                                                                               '"allowedOutsideImage": true,"hasBoundingBox": true,"x":' \
                                           + str(x_right_down) + ',"y":'+ str(y_right_down) + ',"boundingBox": {"width":' \
                                           + str(width) + ', "height": ' + str(height) + ',"left":0,"top":0}}}'


                self.cursor.execute("insert into test_labelinfo(test_lb_id, dl_test_id, studylist_id, file_name, label_info, image_index_number, series_uid, tooltype, del_yn, createdate, ruserid) "
                                    "values(test_labelinfo_test_lb_id_seq.nextval, :dl_test_id, :studylist_id, :file_name, :label_info, :image_index_number, :series_uid, :tooltype, :del_yn, :createdate, :ruserid)"
                                    , [int(self.dl_test_id), self.studylist_id, None, handles, imgindex, self.series_uid, 'rectangleRoi', 'N', datetime.today().strftime("%Y%m%d%H%M%S"), self.ruserid])
                OraDB.dbCommit()
                print('data restored to test_labelinfo')

            print(self.status)
            OraDB.releaseConn()

            del_path = self.file_path + 'DownLoad'
            shutil.rmtree(del_path, ignore_errors=True)
            print('local data deleting finished')


class out_Tester():
    def __init__(self, data_loader, cost_name, act_func, layer_n, ckpt_path, file_path, tr_model_id, dl_test_id, save_path, opt_kwargs={}):
        self.data_loader = data_loader

        if tr_model_id == 1:
            self.ckpt_path = '/home/user01/Javis_dl_system/models/I66/1/Unet.cpkt'

        else:
            # '/home/obsk/Javis_dl_system/models/I66/'
            self.ckpt_path = ckpt_path + str(tr_model_id) + '/' + str(tr_model_id) + '.ckpt'

        self.file_path = file_path
        self.dl_test_id = dl_test_id
        self.save_path = save_path

        # model loader. model id 1 : default model
        if tr_model_id == 1:
            self.net = default_Unet.Unet(cost="mfc", cost_kwargs={"class_weights": [1e-6, 1 - 1e-6]})
        else:
            self.net = Unet.Unet(cost=cost_name, act_func=act_func, layer_n=layer_n,
                                 cost_kwargs={"class_weights": [1e-6, 1 - 1e-6]})

        self.opt_kwargs = opt_kwargs

        # default ai result
        self.status = 'N'

        # # mongo db information
        # self.client = MongoClient('mongodb://172.16.52.79:27017')
        # self.db = self.client.ohif_deployment
        # self.ruserid = self.db.dl_test.find_one({"dl_test_id": int(self.dl_test_id)})['ruserid']
        # self.test_labelinfo_collection = self.db.test_labelinfo
        # self.dl_test_collection = self.db.dl_test
        # self.table_indexing_collection = self.db.table_indexing

        # # oracle db information
        self.cursor = OraDB.prepareCursor()
        self.ruserid = self.cursor.execute("select ruserid from dl_test where dl_test_id=:dl_test_id", {'dl_test_id':self.dl_test_id})
        print('tester initialized')

    # x data O, label data X, OUT
    def infer3(self, n_t_iters, b_size, keep_prob=1.0):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.ckpt_path)
            print('model restored')

            target_img_number_set = ''

            for _ in range(n_t_iters):
                positions = []
                for up_down in range(2):
                    for crop_number in range(4):
                        x, files_list, files_list_down, he, wi = self.data_loader.test_load_batch_2(b_size, up_down, crop_number)  # testing=False

                        predictions = sess.run(self.net.predict, feed_dict={self.net.x: x, self.net.training: False, self.net.keep_prob: keep_prob})

                        if crop_number == 0:
                            x_offset, y_offset = 0, 0
                        elif crop_number == 1:
                            x_offset, y_offset = 44, 0
                        elif crop_number == 2:
                            x_offset, y_offset = 0, 44
                        else:
                            x_offset, y_offset = 44, 44

                        batch_size, nz, ny, nx, _ = np.shape(predictions)
                        for index in range(batch_size):
                            files, offset = files_list[index]

                            for z_index in range(nz):
                                if offset == 0 or 128 <= offset + z_index:
                                    if offset == 0:
                                        file_path = files[z_index]
                                        file_down_path = files_list_down[z_index]

                                    else:
                                        file_path = files[offset + z_index]
                                        file_down_path = files_list_down[offset + z_index]

                                    image = cv2.imread(file_down_path)

                                    if image is None:
                                        image = np.zeros([300, 300, 3])

                                    roi = image[y_offset: y_offset + 256, x_offset: x_offset + 256]
                                    b_prediction = predictions[index, z_index, ..., 0] > 0.5  # 255
                                    b_image = cv2.bitwise_and(roi, roi, mask=(~b_prediction * 255).astype(
                                        np.uint8))  # np.expand_dims
                                    f_image = np.stack([np.zeros([ny, nx]), np.zeros([ny, nx]), b_prediction * 255],
                                                       axis=-1)  # np.tile np.append
                                    merge = b_image + f_image
                                    # merge = np.hstack([image, label, prediction])
                                    image[y_offset: y_offset + 256, x_offset: x_offset + 256] = merge  # f_image
                                    cv2.imwrite(file_down_path, image)

                                    if crop_number == 3:
                                        position_t = cv2.imread(file_down_path)
                                        position = position_t[:, :, 2]
                                        position = cv2.resize(position, (wi, he), interpolation=cv2.INTER_AREA)

                                        _, position = cv2.threshold(position, 127, 255, cv2.THRESH_BINARY)

                                        num_labels, markers, state, cent = cv2.connectedComponentsWithStats(position)

                                        if num_labels != 1:
                                            self.status = 'Y'
                                            for idx in range(1, num_labels):
                                                x, y, w, h, size = state[idx]
                                                infor_position = [z_index, w, h, x, y]
                                                positions.append(infor_position)
                                        roi = cv2.imread(file_path)

                                        b_image = cv2.bitwise_and(roi, roi, mask=~position.astype(np.uint8))
                                        f_image = np.stack([np.zeros([he, wi]), np.zeros([he, wi]), position], axis=-1)
                                        merge = b_image + f_image
                                        # print('file_down_path', file_down_path)
                                        cv2.imwrite(file_down_path, merge)

                # positions = [ [imageindex, width, height, x_start, y_start] ]
                print(positions)
                for img_info in positions:
                    target_img_number_set += str(img_info[0]+1) + '_'

                # print(positions)

                # ai result
                self.cursor.execute("update dl_test set ai_result=:ai_result where dl_test_id = :dl_test_id", {'ai_result':self.status, 'dl_test_id':self.dl_test_id})
                OraDB.dbCommit()

                self.cursor.execute("update op_test set file_path=:file_path where dl_test_id=:dl_test_id", {'file_path':self.save_path, 'dl_test_id':self.dl_test_id})
                OraDB.dbCommit()

                self.cursor.execute("update op_test set valx1=:valx1 where dl_test_id=:dl_test_id", {'valx1':target_img_number_set[:-1], 'dl_test_id':self.dl_test_id})

                print(self.status)
                OraDB.releaseConn()

                print('data copying from local to nas')
                down_path = self.file_path.replace('/upload', '/download') + '/x'
                # temp_path = nas_path.replace('/medimg/', '/home/Javis_dl_system/data_temp/')
                nas_down_path = down_path.replace('/home/user01/Javis_dl_system/data_temp/', '/medimg/')
                chmod_path = nas_down_path[:-2]
                data_mover.on_copytree(down_path, nas_down_path, chmod_path)
                print('data copy finished')

                print('data deleting from local')
                del_path = self.file_path[:self.file_path.find('analysis')+8]
                shutil.rmtree(del_path, ignore_errors=True)
                print('local data deleting finished')
