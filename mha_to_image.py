import re
import skimage.io as io
import scipy.misc as sm
import os

file_list_path = 'E:\\tensorflowdata\\BRATS2015\\'

file_list = []

save_path = 'E:\\tensorflowdata\\BRATS\\'

for (path, dir, files) in os.walk(file_list_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.mha':
            img = io.imread(path+'\\'+filename, plugin='simpleitk')
            for i in range(len(img)):
                path2 = re.sub('2015', '', path)
                filename2 = re.sub('.mha', '', filename)
                dirname = path2 + '\\' + filename2
                print(dirname)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                filename3 = path2 + '\\' + filename2 + '\\' + str(i) + '.jpg'
                if 'OT.' in filename3:
                    img[i] = img[i]*255
                sm.imsave(filename3, img[i])
