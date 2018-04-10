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
    def data_list_load(self, path, mode):
        # augmentation option number generator
        # random_number_for_flip = int(np.random.randint(0, 3, size=1)[0])
        # random_number_for_rotate = int(np.random.randint(0, 4, size=1)[0])

        if mode == 'train':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            label_list = []
    
            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
                        if '/x' in dir_path and '/x_filtered' not in dir_path:
                            if len(os.listdir(dir_path)) != 0:

                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                # y_path_list = [path.replace('/x/', '/x_filtered/') for path in y_path_list]
                                y_path_list = [path.replace('/x/', '/y/') for path in y_path_list]

    
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
            img1 = img1.reshape([self.img_size, self.img_size, 1])
            img2 = img2.reshape([self.img_size, self.img_size, 1])
            img = np.concatenate((img1, img2), axis=2)
            # print(img)
            data.append(img)

        return np.array(data).reshape([-1, self.img_size, self.img_size, 2])


    def _idle(self, img):
        return img

    def _flip_updown(self, img):
        # 이미지 반전,  0:상하, 1 : 좌우
        img = cv2.flip(img, 0)
        return img

    def _flip_leftright(self, img):
        img = cv2.flip(img, 1)
        return img

    def _rotate_90(self, img):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 90, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        return img

    def _rotate_180(self, img):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 180, 1)
        img = cv2.warpAffine(img, m, (nx, ny))
        return img

    def _rotate_270(self, img):
        ny, nx = np.shape(img)
        m = cv2.getRotationMatrix2D((nx / 2, ny / 2), 270, 1)
        img = cv2.warpAffine(img, m, (ny, nx))
        return img

    def _data_flip(self, img, augment_number):
        case = {
            0: self._idle,
            1: self._flip_updown,
            2: self._flip_leftright,

        }
        return case[augment_number](img)

    def _data_rotation(self, img, augment_number):
        case = {
            0: self._idle,
            1: self._rotate_90,
            2: self._rotate_180,
            3: self._rotate_270
        }
        return case[augment_number](img)