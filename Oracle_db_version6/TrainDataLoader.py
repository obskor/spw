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

    def get(self):
        element = self.list[self.index]
        self.index += 1
        if len(self.list) <= self.index:
            random.shuffle(self.list)
            self.index = 0
        return element


class DataLoader():
    def __init__(self, data_path_list, k_fold, c_size=256, i_channel=1, n_class=2):
        self.train_img_files_list = CircularList()
        self.validation_img_files_list = CircularList()

        self.data_path_list = data_path_list

        self.k_fold = k_fold

        self.get_data(data_path_list=self.data_path_list)

        self.c_size = c_size

        self.i_channel = i_channel
        self.n_class = n_class

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

    def get_data(self, data_path_list):
        index = 0

        for data_path in data_path_list:
            for root, dirs, files in os.walk(data_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if 'x' in dir_path:
                        if len(os.listdir(dir_path)) != 0:
                            x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                            img_files = self._sort_by_number(x_path_list)
                            if index == 0:
                                self.validation_img_files_list += img_files
                                index += 1
                            else:
                                self.train_img_files_list += img_files
                                index += 1
                                if index == self.k_fold:
                                    index = 0

    def _load_file(self, file_path):
        img = cv2.imread(file_path, 0)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        return np.array(img, dtype=np.float32)

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

    def _crop_4(self, img, label):
        img_shape = np.shape(img)
        center_x = img_shape[1] // 2
        crop_x = center_x - (self.c_size // 2)
        center_y = img_shape[0] // 2
        crop_y = center_y - (self.c_size // 2)

        c_img = img[crop_y: crop_y + self.c_size, crop_x: crop_x + self.c_size]
        c_label = label[crop_y: crop_y + self.c_size, crop_x: crop_x + self.c_size]
        return c_img, c_label

    def _random_crop(self, img, label):
        img_shape = np.shape(img)
        rx = np.random.randint(0, img_shape[1] - self.c_size)
        ry = np.random.randint(0, img_shape[0] - self.c_size)
        c_img = img[ry: ry + self.c_size, rx: rx + self.c_size]
        c_label = label[ry: ry + self.c_size, rx: rx + self.c_size]
        return c_img, c_label

    def _crop(self, img, label, crop_number):
        case = {
            0: self._crop_0,
            1: self._crop_1,
            2: self._crop_2,
            3: self._crop_3,
            4: self._crop_4,
            5: self._random_crop
        }
        return case[crop_number](img, label)

    def _pre_process(self, x, y, crop_number):
        p_img = self._normalization(x)
        p_label = self._process_label(y)
        c_img, c_label = self._crop(p_img, p_label, crop_number)
        return c_img, c_label

    def _id(self, img, label):
        return img, label

    def _flip_0(self, img, label):
        img = cv2.flip(img, 0)
        label = cv2.flip(label, 0)
        return img, label

    def _flip_1(self, img, label):
        img = cv2.flip(img, 1)
        label = cv2.flip(label, 1)
        return img, label

    def _rotate_90(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 90, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        label = cv2.warpAffine(label, m, (ny, nx))
        return img, label

    def _rotate_180(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 180, 1)
        img = cv2.warpAffine(img, m, (nx, ny))
        label = cv2.warpAffine(label, m, (nx, ny))
        return img, label

    def _rotate_270(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 270, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        label = cv2.warpAffine(label, m, (ny, nx))
        return img, label

    def _data_augment(self, img, label, augment_number):
        case = {
            0: self._id,
            1: self._flip_0,
            2: self._flip_1,
            3: self._rotate_90,
            4: self._rotate_180,
            5: self._rotate_270
        }
        return case[augment_number](img, label)

    def load_batch(self, up_down, crop_number, augment_number, training, testing, b_size=1):

        if training is True:
            img_files_list = self.train_img_files_list
        elif training is False:
            img_files_list = self.validation_img_files_list

        x = np.zeros((b_size, 128, self.c_size, self.c_size, self.i_channel))  # max
        y = np.zeros((b_size, 128, self.c_size, self.c_size, self.n_class))

        files_list = []

        for index in range(b_size):
            img_files = img_files_list.get()
            length = len(img_files)

            offset = up_down * (length - 128)

            files_list.append((img_files, offset))

            for z_index in range(128):
                img_file = img_files[offset + z_index]
                label_file = img_file.replace('x', 'y')
                img = self._load_file(img_file)
                label = self._load_file(label_file)

                p_img, p_label = self._pre_process(img, label, crop_number)
                a_img, a_label = self._data_augment(p_img, p_label, augment_number)

                x[index, z_index] = np.reshape(a_img, (self.c_size, self.c_size, self.i_channel))
                y[index, z_index] = a_label

        return (x, y) if not testing else (x, y, files_list)

