import cv2
import numpy as np

file_down_path = 'C:\\Users\\YEL\\Documents\\LYE\\workspace\\cerebral_aneurysm\\spw\\dataset_test\\result' \
                 '\\180412-twostep-dice-label_only-batch_on-re_conv_lrx10_first_step\\1\\img\\result_only\\MR000000.png'

positions= []

position_t = cv2.imread(file_down_path)
position = position_t[:, :, 2]
_, position = cv2.threshold(position, 127, 255, cv2.THRESH_BINARY)
num_labels, markers, state, cent = cv2.connectedComponentsWithStats(position)
if num_labels != 1:
    status = 'Y'
    for idx in range(1, num_labels):
        x, y, w, h, size = state[idx]
        infor_position = [w, h, x, y]
        positions.append(infor_position)

roi = cv2.imread(file_path)
b_image = cv2.bitwise_and(roi, roi, mask=~position.astype(np.uint8))
f_image = np.stack([np.zeros([he, wi]), np.zeros([he, wi]), position], axis=-1)
merge = b_image + f_image