####################################################################################################
# Created by LYE, 2018-03-26.
# 학습된 First Step Model로 MRI 뇌 사진에서 혈관 위치를 뽑아낸 뒤,
# 혈관에서 뇌동맥류를 찾는 Second Step Model 학습에 필요한 X 데이터와 Y 데이터를 생성한다.
####################################################################################################
import os
import cv2
import unet_first_step_model
import tensorflow as tf
import numpy as np
import traceback
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LOG_ENABLE = False


class Interlude:
    def __init__(self, data_path, model_path, depth, option_name, n_class=2, batch_size=20, img_size=256):
        self.img_size = img_size
        self.batch_size = batch_size
        self.depth = depth
        self.option_name = option_name

        # load data directories
        self.paths = data_path
        self.img_dir_list = self._test_data_load()

        # load model
        self.model_path = model_path
        self.model = unet_first_step_model.Model(batch_norm_mode='off', depth=self.depth, img_size=img_size,
                                                 n_channel=1, n_class=n_class, batch_size=self.batch_size)

    ###
    # * 일련의 테스트 과정을 총괄하는 구동 함수.
    # First Step의 결과값으로 Second Step의 X 및 Y로 사용할 이미지를 생성한다.
    # First Step : 뇌 MRI 사진에서 혈관을 찾는 모델을 학습시킨다.
    # Second Step : 혈관 이미지에서 뇌동맥류를 찾는 모델을 학습시킨다.
    ###
    def process(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.model_path)

        for idx, img_dir in enumerate(self.img_dir_list):
            print('Labeling Directory {0}...({1}/{2})'.format(img_dir, idx, len(self.img_dir_list)))

            # First Step X (뇌 MRI 이미지)
            first_X_dir_full_path = os.path.join(img_dir, 'x')
            # Second Step Y의 근간이 되는 뇌동맥류 라벨 데이터
            second_pre_Y_dir_full_path = os.path.join(img_dir, 'y')
            # First Step Y를 저장할 경로
            first_Y_dir_full_path = os.path.join(img_dir, self.option_name + '/first_output')
            # Second Step X를 저장할 경로
            second_X_dir_full_path = os.path.join(img_dir, self.option_name + '/second_x')
            # Second Step Y를 저장할 경로
            second_Y_dir_full_path = os.path.join(img_dir, self.option_name + '/second_y')

            # First Step X List
            first_X_list = os.listdir(first_X_dir_full_path)

            while first_X_list:
                # 디렉토리 내의 이미지 파일을 batch size 단위로 잘라서 사용
                img_files = first_X_list[:self.batch_size]
                del first_X_list[:self.batch_size]

                x = self._load_img(first_X_dir_full_path, img_files)
                pre_y = self._load_img(second_pre_Y_dir_full_path, img_files)

                self.model.batch_size = len(x)

                try:
                    labels = self._predict(sess, x)

                except Exception as e:
                    logging.error(traceback.format_exc())
                    print('error file at : ', first_X_list, img_files)
                    continue

                for idx, label in enumerate(labels):
                    # MRI 사진 위에 라벨을 오버레이한 이미지 한 장씩 보기
                    # self._show_label_overlapped_img(label, first_X_dir_full_path, img_files[idx])

                    # first model의 output을 one-hot화
                    label = cv2.threshold(label * 255, 120, 255, cv2.THRESH_BINARY)

                    # 혈관 라벨 데이터 저장
                    self._save_first_Y(os.path.join(first_Y_dir_full_path, img_files[idx]), label)

                    # second step X 데이터(혈관부분 원본 값) 생성
                    self._create_second_X(os.path.join(second_X_dir_full_path, img_files[idx]), label, x[idx])

                    # second step Y 데이터(혈관부분 질병 값) 생성
                    self._create_second_Y(os.path.join(second_Y_dir_full_path, img_files[idx]), label, pre_y[idx])

            print('Complete!')
        print('Labeling Complete!')

    ###
    # * 원본 이미지 위에 결과 라벨을 붉은 색으로 겹쳐서 보여준다.
    # label : 위에 겹칠 결과값
    # img_path, img_name : 원본 이미지가 존재하는 경로 및 파일명
    # alpha : 투명도(0~1, 높을수록 진하게)
    ###
    def _show_label_overlapped_img(self, label, img_path, img_name, alpha=0.2):
        label_red = self._gray_to_color(label, 'RED')
        label_red = cv2.convertScaleAbs(label_red * 255)

        # 원본 이미지를 3채널로 불러와서 img_size로 resizing
        img_original = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_COLOR)
        img_original = cv2.resize(img_original, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img_res = img_original.copy()
        cv2.addWeighted(img_original, alpha, label_red, 1 - alpha, 0, img_res)

        cv2.imshow('original', img_original)
        cv2.imshow('label_overlay', img_res)
        cv2.waitKey()

    ###
    # * 경로 내의 이미지 파일들을 모두 읽어들여 반환하는 함수.
    # full_path = file들이 들어있는 전체 경로
    # files : full_path 안의 file 이름들을 담은 리스트
    # return : (batch size, img size, img size, 1)
    ###
    def _load_img(self, full_path, files):
        data = []
        for filename in files:
            img = cv2.imread(os.path.join(full_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            data.append(img)
        return np.array(data).reshape([-1, self.img_size, self.img_size, 1])

    ###
    # * 모델 사용을 위한 사전 작업
    # 1. 경로 내에서 'x' 디렉토리를 가지고 있는 모든 디렉토리를 리스트에 담아 반환한다.
    # 2. 결과 이미지를 담을 x_result 디렉토리가 없으면 미리 생성해둔다.
    # return : 경로 내에서 직속으로 'x' 디렉토리를 가진 모든 상위 디렉토리의 full path
    ###
    def _test_data_load(self):
        dir_list = []

        for path in self.paths:
            subroots = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
            for subroot in subroots:
                # img가 서브디렉토리로 있어서 경로에 추가
                subroot = os.path.join(subroot, 'img')
                if 'x' in os.listdir(os.path.join(path, subroot)):
                    if 'first_output' not in os.listdir(os.path.join(path, subroot)):
                        os.mkdir(os.path.join(path, subroot, self.option_name, 'first_output'))
                    if 'second_x' not in os.listdir(os.path.join(path, subroot)):
                        os.mkdir(os.path.join(path, subroot, self.option_name, 'second_x'))
                    if 'second_y' not in os.listdir(os.path.join(path, subroot)):
                        os.mkdir(os.path.join(path, subroot, self.option_name, 'second_y'))

                dir_list.append(os.path.join(path, subroot))

        return dir_list

    ###
    # * 1채널 회색 이미지를 3채널 유색 이미지로 변환
    # img_gray : 1채널 이미지 데이터 (img_size, img_size, 1)
    # return.shape : (img_size, img_size, 3)
    ###
    def _gray_to_color(self, img_gray, color):
        _ = np.zeros(img_gray.shape)
        r = img_gray

        if color == 'RED':
            return np.concatenate((_, _, r), axis=2)
        elif color == 'BLUE':
            return np.concatenate((r, _, _), axis=2)
        elif color == 'GREEN':
            return np.concatenate((_, r, _), axis=2)

    ###
    # * 모델의 출력 y를 반환
    ###
    def _predict(self, sess, x):
        feed_dict = {self.model.X: x, self.model.drop_rate: 0, self.model.training: False}
        predicted_result = sess.run(self.model.foreground_predicted, feed_dict=feed_dict)

        return predicted_result

    ###
    # * 이미지 저장
    # 255를 곱하여 값 범위를 0~1에서 0~255로 바꾸어 저장한다.
    ###
    def _save_first_Y(self, path, img):
        cv2.imwrite(path, img)

    def _create_second_X(self, path, label, X):
        cv2.imwrite(path, label * X)

    def _create_second_Y(self, path, label, pre_Y):
        cv2.imwrite(path, label * pre_Y)


def u_log(*args):
    if LOG_ENABLE:
        for arg in args:
            print(arg, end='')
        print()


if __name__ == "__main__":
    tester = Interlude(img_size=256, data_path=['./new_data/'], model_path='./models/Unet.ckpt', batch_size=22)
    tester.process()