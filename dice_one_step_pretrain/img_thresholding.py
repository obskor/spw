###
# 2018.03.19 LYE
# MRI 뇌 사진에서 혈관만 뽑아내기 위한 threshold 작업
###
import cv2
import os


def paint_to_black(img, rate):
    img_res = img.copy()
    x, y = img_res.shape
    for i in range(x):
        for j in range(y):
            if i < x*rate or i > x*(1-rate) or j < x*rate or j > x*(1-rate):
                img_res[i][j] = 0
    return img_res


def save_img(img, path, filename):
    full_path = os.path.join(path, 'img', 'first_y')
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    try:
        cv2.imwrite(os.path.join(full_path, filename), img)
    except:
        print('fail: ' + filename)


def threshold(img, thresh, binary):
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] < thresh:
                img[i][j] = 0
            else:
                if binary:
                    img[i][j] = 255
    return img


def img_threshold(roots):
    for root in roots:
        # subroot ex) 1.2.410.2000010.82.2291.1254472141230004.99572
        subroots = os.listdir(root)
        subroots.sort(key=float)
        for subroot in subroots:
            print('Filtering directory {0}...'.format(subroot))
            imgdir = os.path.join(root, subroot)
            imgdir_x = os.path.join(imgdir, 'img', 'x')

            for dirpath, dirnames, filenames in os.walk(imgdir_x):
                filenames.sort()
                for filename in filenames:
                    # print(filename)
                    img_x = cv2.imread(os.path.join(imgdir_x, filename), cv2.IMREAD_GRAYSCALE)
                    img_x_norm = cv2.normalize(img_x, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                    # threshold
                    # binary 추출
                    # img_thresholded = cv2.threshold(img_x_norm, 120, 255, cv2.THRESH_BINARY)
                    # 원본 추출
                    img_thresholded = threshold(img_x_norm, 120, True)
                    img_thresholded_blacked = paint_to_black(img_thresholded, 0.15)

                    save_img(img_thresholded_blacked, imgdir, filename)

if __name__ == "__main__":
    img_threshold("./new_data/")


# import cv2
# import os
# from matplotlib import pyplot as plt
# from PIL import Image
#
# #####
# #   경로 내의 x/y 디렉토리 각각의 이미지 파일을 매칭해서 여는 코드.
# #   ex) 루프 첫 바퀴에 x\001.png와 y\001.png를 함께 로드
# #####
# def search(root):
#     # subroot ex) 1.2.410.2000010.82.2291.1254472141230004.99572
#     subroots = os.listdir(root)
#     for subroot in subroots:
#         imgdir_x = os.path.join(root, subroot, 'x')
#         imgdir_y = os.path.join(root, subroot, 'y')
#
#         for dirpath, dirnames, filenames in os.walk(imgdir_x):
#             for filename in filenames:
#                 img_x = cv2.imread(os.path.join(imgdir_x, filename), cv2.IMREAD_GRAYSCALE)
#                 img_y = cv2.imread(os.path.join(imgdir_y, filename), cv2.IMREAD_GRAYSCALE)
#
#                 cv2.imshow('1', img_x)
#                 cv2.waitKey()
#                 cv2.imshow('2', img_y)
#
#                 # fig = plt.figure(figsize=(1,2))
#                 # fig.add_subplot(1, 2, 1)
#                 # plt.xticks([])
#                 # plt.yticks([])
#                 # plt.imshow(img_x)
#                 # fig.add_subplot(1, 2, 2)
#                 # plt.imshow(img_y)
#                 input("")
#                 print(filename)
#             # print('dirpath : ' + dirpath)
#             # print(dirnames)
#             # print(filenames)
#
# search("./data/")
