# -*- coding: utf-8 -*-

import os
import dicom
import mritopng
import time
import shutil


class Converter:
    def __init__(self, path):
        self.dlserver_path = path
        self.nas_path = self.dlserver_path.replace('/home/user01/Javis_dl_system/data_temp/','/medimg/')

    def _convert(self):
        print('open data converving start now')
        tot_convert_time = 0

        for root, dirs, files in os.walk(self.dlserver_path):

            for file in files:
                if '.dcm' in file:
                    dcm_path=os.path.join(root, file)
                    print('dcm path', dcm_path)
                    dicom_read = dicom.read_file(dcm_path)

                    try:

                        if '3D' in dicom_read.SeriesDescription:
                            x_save_path = dcm_path[:dcm_path.find('/upload')+7] + '/x'
                            print('x_save_path', x_save_path)

                            if not os.path.exists(x_save_path):
                                os.makedirs(x_save_path)

                            convert_s_time = time.time()
                            print('save name', x_save_path + '/' + dcm_path[dcm_path.find('/upload')+8:-4] + '.jpg')
                            mritopng.convert_file(dcm_path, x_save_path + '/' + dcm_path[dcm_path.find('/upload')+8:-4] + '.jpg')
                            convert_e_time = time.time()
                            tot_convert_time += convert_e_time - convert_s_time

                    except AttributeError:

                        pass

        print('converting Finished. Converting time', tot_convert_time)

    def _dlserver_to_nas(self):
        shutil.move(self.dlserver_path, self.nas_path)



