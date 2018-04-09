import numpy as np
import cv2
import os


def rm_black_img(roots):
    for root in roots:
        # subroot ex) 1.2.410.2000010.82.2291.1254472141230004.99572
        subroots = os.listdir(root)
        subroots.sort(key=float)
        # print(subroots)
        for subroot in subroots:
            print('Filtering directory {0}...'.format(subroot))
            imgdir = os.path.join(root, subroot)
            imgdir_x = os.path.join(imgdir, 'img', 'x')
            imgdir_y = os.path.join(imgdir, 'img', 'y')

            for dirpath, dirnames, filenames in os.walk(imgdir_y):
                print(dirpath, dirnames)
                filenames.sort()
                for filename in filenames:
                    # print(filename)
                    img_y = cv2.imread(os.path.join(imgdir_y, filename), 0)

                    if cv2.countNonZero(img_y) == 0:
                        print('black img : ', filename)
                        os.remove(os.path.join(imgdir_y, filename))
                        os.remove(os.path.join(imgdir_x, filename))
                    else:
                        print('colored img: ', filename)


rm_black_img(["./new_data/"])
