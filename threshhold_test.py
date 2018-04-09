import cv2
import os

# 3x3 filter 적용 (중심 가중치 2, 주변 가중치 1)
def average_filter(img):
    img_res = img.copy()
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         try:
    #             img_res[i][j] = np.floor((img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1] \
    #                             + img[i][j - 1] + 5*img[i][j] + img[i - 1][j + 1] \
    #                             + img[i + 1][j - 1] + img[i + 1][j] + img[i + 1][j + 1]) / 13)
    #         except:
    #             img_res[i][j] = 0

    return img_res

# 이미지의 상하좌우 일정 비율을 검은색으로 만든다.
def paint_to_black(img, rate):
    img_res = img.copy()
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if i < x*rate or i > x*(1-rate) or j < x*rate or j > x*(1-rate):
                img_res[i][j] = 0
    return img_res


def save_img(img, path, filename):
    try:
        full_path = os.path.join(path, 'x_filtered')
        os.mkdir(full_path)
        cv2.imwrite(os.path.join(full_path, filename), img)
    except:
        pass

def threshold(img, thresh):
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] < thresh:
                img[i][j] = 0
    return img


def search(root):
    # subroot ex) 1.2.410.2000010.82.2291.1254472141230004.99572
    subroots = os.listdir(root)
    for subroot in subroots:
        imgdir = os.path.join(root, subroot)
        imgdir_x = os.path.join(imgdir, 'x')
        imgdir_y = os.path.join(imgdir, 'y')

        for dirpath, dirnames, filenames in os.walk(imgdir_x):
            filename = '083.png'
            img_x = cv2.imread(os.path.join(imgdir_x, filename), cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(os.path.join(imgdir_y, filename), cv2.IMREAD_GRAYSCALE)
            img_x_norm_minmax = cv2.normalize(img_x, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            img_x_filtered_norm = average_filter(img_x_norm_minmax)

            # threshold
            # 120의 경우 80보다 혈관이 많이 잘려나간다.
            _, img_thresholded_80_filtered = cv2.threshold(img_x_filtered_norm, 75, 255, cv2.THRESH_BINARY)
            _, img_thresholded_80 = cv2.threshold(img_x_norm_minmax, 80, 255, cv2.THRESH_BINARY)
            img_thresholded_80_blacked = paint_to_black(img_thresholded_80, 0.15)
            _, img_thresholded_120 = cv2.threshold(img_x_norm_minmax, 120, 255, cv2.THRESH_BINARY)
            img_thresholded_120_blacked = paint_to_black(img_thresholded_120, 0.15)

            img_thresholded_80_self_minmax = threshold(img_x_norm_minmax, 80)

            # 80 threshold를 준 뒤에 다시 한번 가공
            img_thresholded_mean = cv2.adaptiveThreshold(img_thresholded_80_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                         cv2.THRESH_BINARY, 3, 2)
            img_thresholded_gaussian = cv2.adaptiveThreshold(img_thresholded_80_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 3, 2)

            # overlay label on original image
            img_overlayed_x = img_x_norm_minmax.copy()
            img_overlayed_80_filtered = img_thresholded_80_filtered.copy()
            img_overlayed_80 = img_thresholded_80_blacked.copy()
            img_overlayed_120 = img_thresholded_120.copy()


            img_overlayed_mean = img_thresholded_mean.copy()
            img_overlayed_gaussian = img_thresholded_gaussian.copy()

            img_overlay_y = img_y.copy()
            alpha = 0.2
            cv2.putText(img_overlay_y, "Alpha = {}".format(alpha),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            cv2.addWeighted(img_overlay_y, alpha, img_x_norm_minmax, 1 - alpha, 0, img_overlayed_x)
            cv2.addWeighted(img_overlay_y, alpha, img_thresholded_80_filtered, 1 - alpha, 0, img_overlayed_80_filtered)
            cv2.addWeighted(img_overlay_y, alpha, img_thresholded_80_blacked, 1 - alpha, 0, img_overlayed_80)
            cv2.addWeighted(img_overlay_y, alpha, img_thresholded_120, 1 - alpha, 0, img_overlayed_120)
            cv2.addWeighted(img_overlay_y, alpha, img_thresholded_mean, 1 - alpha, 0, img_overlayed_mean)
            cv2.addWeighted(img_overlay_y, alpha, img_thresholded_gaussian, 1 - alpha, 0, img_overlayed_gaussian)

            cv2.imshow('original', img_x)
            cv2.waitKey()
            cv2.imshow('norm_minmax', img_x_norm_minmax)
            cv2.waitKey()
            # cv2.imshow('maxnorm', img_x_filtered_norm)
            # cv2.waitKey()
            # cv2.imshow('filtered_80', img_thresholded_80)
            # cv2.waitKey()
            cv2.imshow('filtered_80_self_minmax', img_thresholded_80_self_minmax)
            cv2.waitKey()
            cv2.imshow('filtered_80_blacked', img_thresholded_80_blacked)
            cv2.waitKey()
            cv2.imshow('80', img_thresholded_80_blacked)
            cv2.waitKey()
            cv2.imshow('120', img_thresholded_120)
            cv2.waitKey()
            cv2.imshow('12_blacked', img_thresholded_120_blacked)
            # cv2.waitKey()
            # cv2.imshow('4', img_thresholded_mean)
            # cv2.waitKey()
            # cv2.imshow('overlayed_x', img_overlayed_x)
            # cv2.waitKey()
            # cv2.imshow('overlayed_80_filtered', img_overlayed_80_filtered)
            cv2.waitKey()
            cv2.imshow('overlayed_80', img_overlayed_80)
            cv2.waitKey()
            cv2.imshow('overlayed_120', img_overlayed_120)
            cv2.waitKey()
            # cv2.imshow('overlayed_mean', img_overlayed_mean)
            # cv2.waitKey()
            # cv2.imshow('overlayed_gaussian', img_overlayed_gaussian)
            # cv2.waitKey()

            save_img(img_thresholded_80_blacked, imgdir, filename)
            print('complete!')
            break
        break

search("./data/")
