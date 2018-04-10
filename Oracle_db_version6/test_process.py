# -*- coding: utf-8 -*-

"""
Sub Processor Module. Made by BJH. OBS Korea
"""


import TestDataLoader
from Test import in_Tester, out_Tester
import time
from pymongo import MongoClient
import Open_Dicom_converter
import os
import shutil
import data_mover
import tensorflow as tf
from collections import deque
from Oracle_connector import OraDB


class test_proc:
    def __init__(self, dl_test_id, test_type, tr_model_id, studylist_id, file_path):
        self.dl_test_id = dl_test_id
        self.test_type = test_type
        self.tr_model_id = tr_model_id
        self.studylist_id = studylist_id
        print('test_proc', self.studylist_id)
        self.file_path = file_path

    def in_test_run(self, **kwargs):
        dl_test_id = kwargs['dl_test_id']
        test_type = kwargs['test_type']
        tr_model_id = kwargs['tr_model_id']
        studylist_id = kwargs['studylist_id']
        file_path = kwargs['file_path']
        local_path = file_path

        ckpt_path = '/home/user01/Javis_dl_system/models/I66/'

        # 불러와야될 tr_model의 설정값을 가져옴
        cur = OraDB.prepareCursor()
        cur.execute("select layer_cnt, loss_info, activation_info from training where tr_model_id=:tr_model_id", {'tr_model_id': tr_model_id})
        for row in cur:
            layer_n, cost_name, act_func = list(row)
        OraDB.releaseConn()

        # 공공 구분. 공공이면 파일 패스에서 다이컴 컨버팅 실행
        temp_path = local_path.replace('/medimg/', '/home/user01/Javis_dl_system/data_temp/')

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        data_mover.nas_to_dlserver(local_path, temp_path)

        data_loader = TestDataLoader.DataLoader(temp_path, test_type)
        tester = in_Tester(file_path = temp_path, ckpt_path = ckpt_path,data_loader=data_loader, act_func=act_func, cost_name=cost_name, layer_n=layer_n, tr_model_id=tr_model_id, dl_test_id=dl_test_id, studylist_id=studylist_id)
        a = data_loader._existence_label()

        if a == 0:
            tester.infer2(1, 1)
            del_path = temp_path + '/img/DownLoad'
            shutil.rmtree(del_path, ignore_errors=True)
        else:
            tester.infer1(1, 1)
            del_path = temp_path + '/img/DownLoad'
            shutil.rmtree(del_path, ignore_errors=True)

    def out_test_run(self, **kwargs):
        dl_test_id = kwargs['dl_test_id']
        test_type = kwargs['test_type']
        tr_model_id = kwargs['tr_model_id']
        file_path = kwargs['file_path']

        local_path = file_path.replace('/public_open', '/medimg/public_open')

        # /public_open/analysis/PCB64CAED/upload
        open_save_path = file_path.replace('/upload', '/download/x')

        ckpt_path = '/home/user01/Javis_dl_system/models/I66/'

        # 불러와야될 tr_model의 설정값을 가져옴
        cur = OraDB.prepareCursor()
        cur.execute("select layer_cnt, loss_info, activation_info from training where tr_model_id=:tr_model_id", {'tr_model_id': tr_model_id})
        for row in cur:
            layer_n, cost_name, act_func = list(row)
        OraDB.releaseConn()

        print('file path', file_path)
        # 공공 구분. 공공이면 파일 패스에서 다이컴 컨버팅 실행
        print('local path', local_path)
        temp_path = local_path.replace('/medimg/public_open', '/home/user01/Javis_dl_system/data_temp/public_open')

        print('temp path', temp_path)

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        down_path = temp_path.replace('/upload', '/download') + '/x'
        nas_down_path = down_path.replace('/home/user01/Javis_dl_system/data_temp/public_open', '/medimg/public_open')

        print('nas down path', nas_down_path)
        if not os.path.exists(nas_down_path):
            os.makedirs(nas_down_path)

        data_mover.nas_to_dlserver(local_path, temp_path)
        dcm_converting = Open_Dicom_converter.Converter(temp_path)
        dcm_converting._convert()

        data_loader = TestDataLoader.DataLoader(temp_path, test_type)
        tester = out_Tester(file_path=temp_path, ckpt_path=ckpt_path, data_loader=data_loader, act_func=act_func,
                            cost_name=cost_name, layer_n=layer_n, tr_model_id=tr_model_id, dl_test_id=dl_test_id, save_path=open_save_path)

        tester.infer3(1, 1)

    def ts_main_process(self):
        # DB 체크 후 process 정보 가져옴
        time.sleep(3)

        # if cur.execute("select seq where index_name = test_lb_id") is None:
        #     cur.execute("update table_indexing set seq = 0 where index_name = test_lb_id")
        #     cur.commit()

        # if self.test_type == 'in':
        #     # 테스트 시작
        #     print('test start')
        #     db.dl_test.find_and_modify(query={'dl_test_id': int(self.dl_test_id)}, update={"$set": {'dl_status': "running"}}, upsert=False, full_response=True)
        #     self.in_test_run(dl_test_id = self.dl_test_id, test_type = self.test_type, tr_model_id = self.tr_model_id, studylist_id = int(self.studylist_id), file_path = self.file_path)
        #     # 테스트가 종료되면 dl_status를 end로 갱신
        #     db.dl_test.find_and_modify(query={'dl_test_id': int(self.dl_test_id)}, update={"$set": {'dl_status': "end"}}, upsert=False, full_response=True)
        #
        # elif self.test_type == 'out':
        #     # 테스트 시작
        #     print('test start')
        #     db.dl_test.find_and_modify(query={'dl_test_id': int(self.dl_test_id)}, update={"$set": {'dl_status': "running"}}, upsert=False, full_response=True)
        #     self.out_test_run(dl_test_id=int(self.dl_test_id), test_type=self.test_type, tr_model_id=self.tr_model_id, file_path=self.file_path)
        #
        #     # 테스트가 종료되면 dl_status를 end로 갱신
        #     db.dl_test.find_and_modify(query={'dl_test_id': int(self.dl_test_id)}, update={"$set": {'dl_status': "end"}}, upsert=False, full_response=True)

        try:
            cur = OraDB.prepareCursor()
            if self.test_type == 'in':
                # 테스트 시작
                print('test start')
                cur.execute("update dl_test set dl_status = running where dl_test_id = :dl_test_id", int(self.dl_test_id))
                OraDB.dbCommit()
                self.in_test_run(dl_test_id = self.dl_test_id, test_type = self.test_type, tr_model_id = self.tr_model_id,
                                 studylist_id = self.studylist_id, file_path = self.file_path)
                # 테스트가 종료되면 dl_status를 end로 갱신
                cur.execute("update dl_test set dl_status = end where dl_test_id = :dl_test_id", int(self.dl_test_id))
                OraDB.dbCommit()
                OraDB.releaseConn()

            elif self.test_type == 'out':
                # 테스트 시작
                print('test start')
                cur.execute("update dl_test set dl_status = running where dl_test_id = :dl_test_id", int(self.dl_test_id))
                OraDB.dbCommit()
                self.out_test_run(dl_test_id=int(self.dl_test_id), test_type=self.test_type, tr_model_id=self.tr_model_id, file_path=self.file_path)

                # 테스트가 종료되면 dl_status를 end로 갱신
                cur.execute("update dl_test set dl_status = end where dl_test_id = :dl_test_id", int(self.dl_test_id))
                OraDB.dbCommit()
                OraDB.releaseConn()

        except:
            cur = OraDB.prepareCursor()
            # 에러 발생시 dl_status를 fail로 갱신
            cur.execute("update dl_test set dl_status = fail where dl_test_id = :dl_test_id", int(self.dl_test_id))
            OraDB.dbCommit()
            OraDB.releaseConn()
            print('Test Process Error Occurred')


# process runner
if __name__ == "__main__":
    time.sleep(3)
    try:
        cur = OraDB.prepareCursor()
        cur.execute("select dl_test_id, test_type, tr_model_id, studylist_id, file_path, dl_status from dl_test")
        for row in cur:
            dl_test_id, test_type, tr_model_id, studylist_id, file_path, dl_status = list(row)
            if dl_status == 'run':
                test = test_proc(dl_test_id=dl_test_id, test_type=test_type,
                                 tr_model_id=tr_model_id,
                                 studylist_id=studylist_id, file_path=file_path)
                test.ts_main_process()

                break
        OraDB.releaseConn()

    except:
        print('Test Error Occured')

    # cursor2 = db.dl_test.find()
    # for row2 in cursor2:
    #     if row2['dl_status'] == 'run':
    #         test = test_proc(dl_test_id=row2['dl_test_id'], test_type=row2['test_type'],
    #                          tr_model_id=row2['tr_model_id'],
    #                          studylist_id=row2['studylist_id'], file_path=row2['file_path'])
    #         test.ts_main_process()





