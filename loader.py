import re
import cv2
import os
import numpy as np
from sys import platform


class DataLoader:
    def __init__(self, img_size):
        self.img_size = img_size

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    # 파일명 번호 순으로 정렬
    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    # 데이터 경로 로더
    def data_list_load(self, path, step):
        # 데이터셋 경로를 담아 둘 빈 리스트 생성
        image_list = []
        label_list = []

        print('platform:', platform)
        if step == 'first':
            if platform.startswith('win'):
                x_dir = '\\x'
                y_dir = '\\first_y'
            elif platform.startswith('linux'):
                x_dir = '/x'
                y_dir = '/first_y'
            else:
                x_dir = '/x'
                y_dir = '/first_y'
        elif step == 'second':
            if platform.startswith('win'):
                x_dir = '\\second_x'
                y_dir = '\\second_y'
            elif platform.startswith('linux'):
                x_dir = '/second_x'
                y_dir = '/second_y'
            else:
                x_dir = '/second_x'
                y_dir = '/second_y'
        elif step == 'one_step':
            if platform.startswith('win'):
                x_dir = '\\x'
                y_dir = '\\y'
            elif platform.startswith('linux'):
                x_dir = '/x'
                y_dir = '/y'
            else:
                x_dir = '/x'
                y_dir = '/y'

        # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
        for data_path in path:
            for root, dirs, files in os.walk(data_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
                    if x_dir in dir_path and y_dir not in dir_path:
                        if len(os.listdir(dir_path)) != 0:
                            x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                            y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                            y_path_list = [path.replace(x_dir, y_dir) for path in y_path_list]

                            images_files = self._sort_by_number(x_path_list)
                            labels_files = self._sort_by_number(y_path_list)

                            for image in images_files:
                                image_list.append(image)
                                # print('xdata:', image)

                            for label in labels_files:
                                label_list.append(label)
                                # print('ydata:', label)

        return image_list, label_list, len(image_list)

    def next_batch(self, data_list, label, idx, batch_size):
        data_list = np.array(data_list)
        label = np.array(label)
    
        batch1 = data_list[idx * batch_size:idx * batch_size + batch_size]
        label2 = label[idx * batch_size:idx * batch_size + batch_size]
    
        index = np.arange(len(batch1))
        np.random.shuffle(index)
        batch1 = batch1[index]
        label2 = label2[index]
    
        return batch1, label2
    
    def data_shuffle(self, data, label):
        data = np.array(data)
        label = np.array(label)
    
        index = np.arange(len(data))
        np.random.shuffle(index)
    
        data = data[index]
        label = label[index]
    
        return data, label
    
    def data_split(self, data, label, val_size):
        data_count = len(data)
        if round(data_count* (val_size / 100)) == 0:
            val_data_cnt = 1
        else:
            val_data_cnt = round(data_count * (val_size / 100))
    
        trainX = data[:-val_data_cnt]
        trainY = label[:-val_data_cnt]
        valX = data[-val_data_cnt:]
        valY = label[-val_data_cnt:]
    
        return trainX, trainY, valX, valY
    
    def read_image_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            data.append(img)
    
        return np.array(data).reshape([-1, self.img_size, self.img_size, 1])

    def read_label_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
            img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]
            img1 = img1.reshape([self.img_size,self.img_size,1])
            img2 = img2.reshape([self.img_size,self.img_size,1])
            img = np.concatenate((img1,img2),axis=2)
            # print(img)
            data.append(img)
    
        return np.array(data).reshape([-1, self.img_size, self.img_size, 2])

    def normalization(self, img_list):
        img_res = []
        for img in img_list:
            img_norm = cv2.normalize(img, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            img_res.append(img_norm)
        return np.array(img_res).reshape([-1, self.img_size, self.img_size, 1])

    def flip_0(self, img, label):
        img = cv2.flip(img, 0)
        label = cv2.flip(label, 0)
        return img, label

    def flip_1(self, img, label):
        img = cv2.flip(img, 1)
        label = cv2.flip(label, 1)
        return img, label

    def rotate_90(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 90, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        label = cv2.warpAffine(label, m, (ny, nx))
        return img, label

    def rotate_180(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 180, 1)
        img = cv2.warpAffine(img, m, (nx, ny))
        label = cv2.warpAffine(label, m, (nx, ny))
        return img, label

    def rotate_270(self, img, label):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 270, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        label = cv2.warpAffine(label, m, (ny, nx))
        return img, label

    def data_augment(self, img, label, augment_number):
        case = {
            0: self._id,
            1: self._flip_0,
            2: self._flip_1,
            3: self._rotate_90,
            4: self._rotate_180,
            5: self._rotate_270
        }
        return case[augment_number](img, label)