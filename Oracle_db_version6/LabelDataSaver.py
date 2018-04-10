# -*- coding: utf-8 -*-

"""
Label Data Saving Module, Made by KBS, BJH. JYJ OBS Korea
"""


import numpy as np
import cv2
import json
import os
from Oracle_connector import OraDB
import re

class DataSaver():
    def __init__(self, path):
        # /medimg/data/I66/DICOM/train/abnormal/I66_mri_AB_00000039
        self.path = path
        # print(self.path)


        # I66_mri_AB_00000007
        if '/abnormal' in self.path:
            start_n = self.path.find('/abnormal')
            leng = len('/abnormal')

        else:
            start_n = self.path.find('/normal')
            leng = len('/normal')

        # print(path[start_n + leng + 1:last_n])
        self.patientName = self.path[start_n + leng + 1:]
        print('DataSaver Initialized')
        self.conn = OraDB.createConn(OraDB.INFO)

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    def saveYdata_labeled(self):
        cur = OraDB.prepareCursor()
        study_uid = cur.execute('select study_uid from ADMIN.studylist where patient_uid=:patient_uid', {'patient_uid':self.patientName})
        labels = cur.execute('select * from ADMIN.ohif_labels where studyinstanceuid=:studyinstanceuid', {'studyinstanceuid':study_uid})

        global w, h
        w = 0
        h = 0

        for row in labels:
            id, measurementdata, userid, studyinstanceuid, tooltype, seriesinstanceuid, \
            currentimageidindex, patientid, patientname, studyindexnumber, \
            imagewidth, imageheight, seriesdescription = list(row)

            if currentimageidindex == float:
                continue

            try:
                # 파일 디렉토리만 포함되어있음 ex - /opt/home/data/I66/NonDICOM/train/abnormal/I66_mri_AB_00000007/x

                # /home/bjh/obsk/v_nas2/I66/DICOM/train/abnormal/I66_mri_AB_00000039
                labelPath = self.path + '/img/y/' + str(currentimageidindex) + '.jpg'
                imageLoc = self.path + '/img/y/'

                if not os.path.exists(imageLoc):
                    os.makedirs(imageLoc)

                w, h = imagewidth, imageheight

                img = cv2.imread(labelPath)
                # print(img)

                if img is None:
                    img = np.zeros((h, w))
                # print(img)

                m_data = json.loads(measurementdata)
                handles = m_data['handles']
                # print(handles)

                if tooltype == 'polygonRoi':
                    pts = []
                    for pt in handles.values():
                        pts.append((int(pt['x']), int(pt['y'])))

                    labeled = cv2.fillConvexPoly(img, np.array(pts), (255, 255, 255))
                    # print(img)
                    cv2.imwrite(labelPath, labeled)
            except:
                continue

        OraDB.releaseConn()

        # self.path = /home/bjh/obsk/v_nas2/I66/DICOM/train/abnormal/I66_mri_AB_00000039
        x_data_path = self.path + '/img/x/'
        x_data_cnt = len(os.listdir(x_data_path))
        x_data = self._sort_by_number(os.listdir(x_data_path))

        for i in range(x_data_cnt):

            if os.path.isfile(self.path + '/img/y/' + str(i+1) + '.jpg') is True:
                os.rename(self.path + '/img/y/' + str(i+1) + '.jpg', x_data_path.replace('/x/', '/y/') + x_data[i])
            else:
                label = np.zeros((h, w))
                file_name = x_data_path.replace('/x/', '/y/') + x_data[i]
                cv2.imwrite(file_name, label)

