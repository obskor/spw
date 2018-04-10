# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import cv2
import tensorflow as tf
import random

class CircularList():

    def __init__(self):
        self.list = []
        self.index = 0

    def __iadd__(self, element):
        self.list.append(element)
        return self

    def get_residual(self):
        return len(self.list) - self.index

    def get(self):
        element = self.list[self.index]
        self.index += 1
        if len(self.list) <= self.index:
            random.shuffle(self.list)
            self.index = 0
        return element

class DataLoader():

    def __init__(self, data_path, test_type, c_size=256, i_channel=1, n_class=2):
        self.test_img_files_list = CircularList()
        self.test_img_files_list_down = CircularList()
        self.index = 0

        print(test_type)
        if data_path is not None:
            if test_type == 'in':
                self.get_data_in(data_path)
            else:
                self.get_data_out(data_path)

        self.c_size = c_size

        self.i_channel = i_channel
        self.n_class = n_class
        print('data_loader initialized')

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    def get_data_out(self, data_path):
        print('get_data_out')
        for root, dirs, files in os.walk(data_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if '/y' in dir_path:
                    self.index += 1
                if '/x' in dir_path:
                    # print('dir_path', dir_path)
                    down_dir_path = dir_path.replace('upload', 'download')
                    if not os.path.exists(down_dir_path):
                        os.makedirs(down_dir_path)
                    x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                    # print('x_path_list', x_path_list)
                    x_path_list_down = [os.path.join(down_dir_path, file) for file in os.listdir(dir_path)]
                    img_files = self._sort_by_number(x_path_list)
                    img_files_down = self._sort_by_number(x_path_list_down)
                    self.test_img_files_list += img_files
                    self.test_img_files_list_down += img_files_down

    def get_data_in(self, data_path):
        print('get_data_in')
        for root, dirs, files in os.walk(data_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if '/y' in dir_path:
                    self.index += 1
                if '/x' in dir_path:
                    # print(dir_path)
                    down_dir_path = dir_path.replace('/x', '/DownLoad/')
                    if not os.path.exists(down_dir_path):
                        os.makedirs(down_dir_path)
                    x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                    # print(x_path_list)
                    x_path_list_down = [os.path.join(down_dir_path, file) for file in os.listdir(dir_path)]
                    img_files = self._sort_by_number(x_path_list)
                    img_files_down = self._sort_by_number(x_path_list_down)
                    self.test_img_files_list += img_files
                    self.test_img_files_list_down += img_files_down




    def _existence_label(self):

        return self.index

    def _load_file(self, file_path):
        img = cv2.imread(file_path, 0)
        a = img.shape
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        return np.array(img, dtype=np.float32), a

    def _normalization(self, img):
        img -= np.amin(img)
        amax = np.amax(img)
        if amax != 0:
            img /= amax
        return img

    def _threshold(self, img):
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def _process_label(self, label):
        t_label = self._threshold(label)
        n_label = self._normalization(t_label)

        if self.n_class == 2:
            l_shape = np.shape(n_label)
            nx = l_shape[1]
            ny = l_shape[0]
            labels = np.zeros((ny, nx, self.n_class))
            labels[..., 0] = n_label
            labels[..., 1] = ~(n_label.astype(np.bool))
            return labels

        return n_label

    def _crop_0(self, img, label):
        c_img = img[0: self.c_size, 0: self.c_size]
        c_label = label[0: self.c_size, 0: self.c_size]
        return c_img, c_label

    def _crop_1(self, img, label):
        img_shape = np.shape(img)
        c_img = img[0: self.c_size, img_shape[1] - self.c_size: img_shape[1]]
        c_label = label[0: self.c_size, img_shape[1] - self.c_size: img_shape[1]]
        return c_img, c_label

    def _crop_2(self, img, label):
        img_shape = np.shape(img)
        c_img = img[img_shape[0] - self.c_size: img_shape[0], 0: self.c_size]
        c_label = label[img_shape[0] - self.c_size: img_shape[0], 0: self.c_size]
        return c_img, c_label

    def _crop_3(self, img, label):
        img_shape = np.shape(img)
        c_img = img[img_shape[0] - self.c_size: img_shape[0], img_shape[1] - self.c_size: img_shape[1]]
        c_label = label[img_shape[0] - self.c_size: img_shape[0], img_shape[1] - self.c_size: img_shape[1]]
        return c_img, c_label


    def _crop(self, img , label, crop_number):
        case = {
            0: self._crop_0,
            1: self._crop_1,
            2: self._crop_2,
            3: self._crop_3,
        }
        return case[crop_number](img, label)

    def _pre_process(self, x, y, crop_number):
        p_img = self._normalization(x)
        p_label = self._process_label(y)
        c_img, c_label=self._crop(p_img, p_label, crop_number)
        return c_img, c_label


    def _crop_0_sole(self, img):
        c_img = img[0: self.c_size, 0: self.c_size]
        return c_img

    def _crop_1_sole(self, img):
        img_shape = np.shape(img)
        c_img = img[0: self.c_size, img_shape[1] - self.c_size: img_shape[1]]
        return c_img

    def _crop_2_sole(self, img):
        img_shape = np.shape(img)
        c_img = img[img_shape[0] - self.c_size: img_shape[0], 0: self.c_size]
        return c_img

    def _crop_3_sole(self, img):
        img_shape = np.shape(img)
        c_img = img[img_shape[0] - self.c_size: img_shape[0], img_shape[1] - self.c_size: img_shape[1]]
        return c_img

    def _crop_sole(self, img, crop_number):
        case = {
            0: self._crop_0_sole,
            1: self._crop_1_sole,
            2: self._crop_2_sole,
            3: self._crop_3_sole,
        }
        return case[crop_number](img)


    def _pre_process_sole(self, x, crop_number):
        p_img = self._normalization(x)
        c_img = self._crop_sole(p_img, crop_number)
        return c_img


    def test_load_batch_1(self, b_size, up_down, crop_number):
        img_files_list = self.test_img_files_list
        img_files_list_down = self.test_img_files_list_down
        b_size = min(b_size, img_files_list.get_residual())

        x = np.zeros((b_size, 128, self.c_size, self.c_size, self.i_channel)) # max
        y = np.zeros((b_size, 128, self.c_size, self.c_size, self.n_class))

        files_list = []

        global img_files_down, h, w
        img_files_down = []
        h = 0
        w = 0

        for index in range(b_size):
            img_files = img_files_list.get()
            img_files_down = img_files_list_down.get()
            length = len(img_files)

            offset = up_down * (length - 128)

            files_list.append((img_files, offset))
            h = 0
            w = 0
            for z_index in range(128):
                img_file = img_files[offset + z_index]
                label_file = img_file.replace('x', 'y')
                img, size_1 = self._load_file(img_file)
                label, size_2 = self._load_file(label_file)

                p_img, p_label = self._pre_process(img, label, crop_number)

                x[index, z_index] = np.reshape(p_img, (self.c_size, self.c_size, self.i_channel))
                y[index, z_index] = p_label
                h = size_1[0]
                w = size_1[1]
        return (x, y, files_list, img_files_down, h ,w)



    def test_load_batch_2(self, b_size, up_down, crop_number):
        img_files_list = self.test_img_files_list
        img_files_list_down = self.test_img_files_list_down
        b_size = min(b_size, img_files_list.get_residual())

        x = np.zeros((b_size, 128, self.c_size, self.c_size, self.i_channel)) # max

        files_list = []

        global img_files_down, h, w
        img_files_down = []
        h = 0
        w = 0

        for index in range(b_size):
            img_files = img_files_list.get()
            img_files_down = img_files_list_down.get()
            length = len(img_files)

            offset = up_down * (length - 128)

            files_list.append((img_files, offset))

            for z_index in range(128):
                img_file = img_files[offset + z_index]
                img, size = self._load_file(img_file)

                p_img = self._pre_process_sole(img, crop_number)
                x[index, z_index] = np.reshape(p_img, (self.c_size, self.c_size, self.i_channel))
                h = size[0]
                w = size[1]

        # print(img_files_down)
        return (x, files_list,img_files_down, h, w)



if __name__ == "__main__":
    data_loader = TestDataLoader("data")
    a = data_loader._existence_label()
    print(a)
    #x, y, files_list, files_list_down, w, h = data_loader.test_load_batch_1(1, 0, 0)
    #print(files_list_down)
    #print(w)
    #print(h)
    #print(y.shape)

    #for i in range(0, label.shape[0]):
    #    for j in range(0, label.shape[1]):
    #        if label[i, j] != 0:
    #            print(label[i, j])