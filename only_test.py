import loader
import tensorflow as tf
from unet_first_step_model import Model
import cv2
import numpy as np


test_data_path = '/home/bjh/Test_dataset/abnormal'

test_data_list, test_data_down_list = loader.data_list_load(test_data_path, mode='test')


with tf.Session() as sess:

    m = Model(sess)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, '/home/bjh/new/2ds/model3/models/Unet.ckpt')

    print("LOAD MODEL")

    for idx, test_data in enumerate(test_data_list):

        test_image = loader.read_image_grey_resized(test_data)

        predicted_result = m.show_result(test_image, 1)

        # 나올때의 shape = [1, 224, 224, 1]
        print(predicted_result.shape)

        G = np.zeros([1,224,224,1])
        B = np.zeros([1,224,224,1])
        R = predicted_result

        predicted_result = np.concatenate((B,G,R),axis=3)
        print(predicted_result.shape)
        predicted_result = np.squeeze(predicted_result)
        print(predicted_result.shape)
        predicted_result = cv2.resize(predicted_result, (256, 256), interpolation=cv2.INTER_AREA)

        tR = test_image
        tG = test_image
        tB = test_image

        test_image = np.concatenate((tB,tG,tR),axis=3)
        test_image = np.squeeze(test_image)
        test_image = cv2.resize(test_image, (256, 256), interpolation=cv2.INTER_AREA)

        test_image = test_image.astype(float)

        predicted_result = predicted_result*255

        print(predicted_result.shape, type(predicted_result))
        print(test_image.shape, type(test_image))

        w = 37

        dst = cv2.addWeighted(predicted_result, float(100 - w) * 0.0001, test_image, float(w) * 0.0001, 0)

        cv2.imwrite(test_data_down_list[idx], dst)

