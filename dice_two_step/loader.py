import re
import cv2
import os
import numpy as np


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

    # path의 하위 디렉토리 리스트 생성 후, x 디렉토리가 있고 x_answer 디렉토리가 없으면 x_answer 디렉토리 생성하고 반환
    # path example : ['./data/', './data2']
    # return example : ['./data/1.2.410', ...]
    def make_output_directory(self, paths):
        dir_list = []
        for path in paths:
            subroots = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
            for subroot in subroots:
                if 'x' in os.listdir(os.path.join(path, subroot)) \
                        and 'x_answer' not in os.listdir(os.path.join(path, subroot)):
                    os.mkdir(os.path.join(path, subroot, 'x_answer'))
                dir_list.append(os.path.join(path, subroot))
        return dir_list


    # 데이터 경로 로더
    def data_list_load(self, path, mode, step, option_name):

        if mode == 'train':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            label_list = []

            if step == 1:
                x_path = '/x'
                y_path = '/first_y'
            elif step == 2:
                x_path = '/' + option_name + '/second_x'
                y_path = '/' + option_name + '/second_y'

            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
                        if x_path in dir_path:
                            if len(os.listdir(dir_path)) != 0:

                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                y_path_list = [path.replace(x_path + '/', y_path + '/') for path in y_path_list]

                                images_files = self._sort_by_number(x_path_list)
                                labels_files = self._sort_by_number(y_path_list)

                                for image in images_files:
                                    image_list.append(image)
                                    # print('xdata:', image)

                                for label in labels_files:
                                    label_list.append(label)
                                    # print('ydata:', label)
    
            return image_list, label_list, len(image_list)
    
        elif mode == 'test':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            down_list = []
    
            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if '/x' in dir_path:
                            if len(os.listdir(dir_path)) != 0:
                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                down_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                down_path_list = [path.replace('/home/bjh/Test_dataset/abnormal', '/home/bjh/new/model_test/result') for path in down_path_list]
    
                                images_files = self._sort_by_number(x_path_list)
                                labels_files = self._sort_by_number(down_path_list)
    
                                for image in images_files:
                                    image_list.append(image)
    
                                for label in labels_files:
                                    down_list.append(label)
                                    # print('ydata:', label)
    
            return image_list, down_list, len(image_list)
    
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

    def _idle(self, img):
        return img

    def flipImage(self, img, option_number):
        # 이미지 반전,  0:상하, 1 : 좌우
        img = cv2.flip(img, option_number)
        return img

    def _flip_leftright(self, img):
        img = cv2.flip(img, 1)
        return img

    def rotateImage(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _data_flip(self, img, augment_number):
        case = {
            0: self._idle(img),
            1: self.flipImage(img, 1),
            2: self.flipImage(img, 0),

        }
        return case[augment_number]

    def _data_rotation(self, img, augment_number):
        case = {
            0: self._idle(img),
            1: self.rotateImage(img, 10),
            2: self.rotateImage(img, -10),
            3: self.rotateImage(img, 20),
            4: self.rotateImage(img, -20)
        }
        return case[augment_number]

    def read_data(self, x_list, y_list, mode):
        if len(x_list) != len(y_list):
            raise AttributeError('The amounts of X and Y data are not equal.')

        else:
            # x
            if type(x_list) != str:
                x_list = x_list
            elif type(x_list) == str:
                x_list = [x_list]

            # y
            if type(y_list) != str:
                y_list = y_list
            elif type(y_list) == str:
                y_list = [y_list]

            x_data = []
            y_data = []

            for i in range(len(x_list)):
                # augmentation option number generator
                # randnum 0 : original
                if mode == 'train':
                    random_number_for_flip = int(np.random.randint(0, 3, size=1)[0])
                    random_number_for_rotate = int(np.random.randint(0, 5, size=1)[0])

                else:
                    random_number_for_flip = 0
                    random_number_for_rotate = 0

                # print(random_number_for_rotate, random_number_for_flip)

                # 원본 X
                x_original = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
                x_original = cv2.resize(x_original, (256, 256), interpolation=cv2.INTER_AREA)

                # 플립/로테이션한 X
                x_rotated_img = self._data_rotation(self._data_flip(x_original, random_number_for_flip), random_number_for_rotate)

                # 원본 Y
                y_original = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
                y_original = cv2.resize(y_original, (256, 256), interpolation=cv2.INTER_AREA)

                # Y 색반전해서 2채널 만들기
                y_original_img = cv2.threshold(y_original, 124, 255, cv2.THRESH_BINARY)[1]
                y_original_bg = cv2.threshold(y_original, 124, 255, cv2.THRESH_BINARY_INV)[1]
                y_original_img = y_original_img.reshape([256, 256, 1])
                y_original_bg = y_original_bg.reshape([256, 256, 1])
                y_ori_2ch = np.concatenate((y_original_img, y_original_bg), axis=2)

                # 플립/로테이션한 Y
                y_rotated_img = self._data_rotation(self._data_flip(y_original, random_number_for_flip), random_number_for_rotate)

                y_rotated_img = cv2.threshold(y_rotated_img, 124, 255, cv2.THRESH_BINARY)[1]
                y_rotated_bg = cv2.threshold(y_rotated_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
                y_rotated_img = y_rotated_img.reshape([256, 256, 1])
                y_rotated_bg = y_rotated_bg.reshape([256, 256, 1])
                y_rotated_img_2ch = np.concatenate((y_rotated_img, y_rotated_bg), axis=2)

                # not augmentation
                # if mode == 'train':
                #     x_data.append(x_original)
                #     y_data.append(y_ori_2ch)

                # augmentation
                x_data.append(x_rotated_img)
                y_data.append(y_rotated_img_2ch)

            return np.array(x_data).reshape([-1, 256, 256, 1]), np.array(y_data).reshape([-1, 256, 256, 2])