"""
Sub Processor Module. Made by BJH. OBS Korea
"""

import time
import os
import data_mover
from random import shuffle
import math
import LabelDataSaver
import TrainDataLoader
from Train import Trainer
from Oracle_connector import OraDB


def train_run(**kwargs):

    cost = kwargs['loss_info']
    optimizer = kwargs['optimizer_info']
    learning_rate = kwargs['learning_rate']
    drop_out_rate = kwargs['drop_out_rate']
    act_func = kwargs['act_func']
    layer_cnt = int(kwargs['layer_cnt'])
    model_id = int(kwargs['tr_model_id'])
    normal_data = kwargs['normal_data']
    abnormal_data = kwargs['abnormal_data']
    validation = kwargs['tr_validation']

    k_fold_list = []
    # validation 수치 변경
    if int(validation) == 2:
        print(validation, type(validation))
        k_fold_list.append(50)
    elif int(validation) == 5:
        print(validation, type(validation))
        k_fold_list.append(20)
    elif int(validation) == 10:
        print(validation, type(validation))
        k_fold_list.append(10)
    elif int(validation) == 15:
        print(validation, type(validation))
        k_fold_list.append(18)
    elif int(validation) == 20:
        print(validation, type(validation))
        k_fold_list.append(5)

    # print('dl_option : ', cost, optimizer, learning_rate, drop_out_rate, act_func, layer_cnt, model_id, validation)

    k_fold = k_fold_list[0]

    def _return_loop_number(string):
        if string != '_':
            under_Bar = string.find('_')
            first_data_n = int(string[:under_Bar])
            last_data_n = int(string[under_Bar + 1:])
            return first_data_n, last_data_n

    # path = '/home/obsk/Javis_dl_system/data/I66'
    # /opt/home/data/I66/DICOM/train/abnormal/I66_mri_AB_00000029
    # /home/obsk/v_nas2/I66/DICOM/train/abnormal/I66_mri_AB_00000029
    # path = 'D:/data/I66'

    # normal data get
    normal_file_path_list = []
    normal_data_chklist = []
    if normal_data != '_':
        normal_start, normal_fin = _return_loop_number(normal_data)
        # normal_length = normal_fin - normal_start + 1

        # client = MongoClient('mongodb://172.16.52.79:27017')
        # db = client.ohif_deployment
        # cursor = db.study_normtraining.find().sort('normtrain_id', pymongo.ASCENDING)

        cur = OraDB.prepareCursor()
        cur.execute("select del_yn, dataid, file_path, normtrain_id from study_normtraining order by normtrain_id asc")

        for row in cur:
            del_yn, dataid, file_path, normtrain_id = list(row)
            if 'del_yn' != 'y' and 'I66' in dataid:
                normal_data_chklist.append(file_path)
        print(normal_data_chklist)
        for idx in range(normal_start-1, normal_fin):
            normal_file_path_list.append(normal_data_chklist[idx])
        print(normal_file_path_list)
        OraDB.releaseConn()

    # abnormal data get
    abnormal_file_path_list = []
    abnormal_data_chklist = []
    if abnormal_data != '_':
        abnormal_start, abnormal_fin = _return_loop_number(abnormal_data)
        # abnormal_length = abnormal_fin - abnormal_start + 1

        # client = MongoClient('mongodb://172.16.52.79:27017')
        # db = client.ohif_deployment
        # cursor = db.study_abnormtraining.find().sort('abnormtrain_id', pymongo.ASCENDING)

        cur = OraDB.prepareCursor()
        cur.execute("select del_yn, dataid, file_path, normtrain_id from study_abnormtraining order by abnormtrain_id asc")

        for row in cur:
            del_yn, dataid, file_path, normtrain_id = list(row)
            if del_yn != 'y' and 'I66' in dataid:
                abnormal_data_chklist.append(file_path)
        print(abnormal_data_chklist)
        for idx in range(abnormal_start-1, abnormal_fin):
            abnormal_file_path_list.append(abnormal_data_chklist[idx])
        print(abnormal_file_path_list)
        OraDB.releaseConn()

    # copy nas normal data to local temp directory
    local_n_datapath_list = []
    for nas_path in normal_file_path_list:
        temp_path = nas_path.replace('/medimg/', '/home/user01/Javis_dl_system/data_temp/')
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        data_mover.nas_to_dlserver(nas_path, temp_path)
        local_n_datapath_list.append(temp_path)

    # copy nas abnormal data to local temp directory
    local_ab_datapath_list = []
    for nas_path in abnormal_file_path_list:
        temp_path = nas_path.replace('/medimg/', '/home/user01/Javis_dl_system/data_temp/')
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        data_mover.nas_to_dlserver(nas_path, temp_path)
        local_ab_datapath_list.append(temp_path)

    tot_datapath_list = local_n_datapath_list + local_ab_datapath_list
    # print('tot_data_id_list : ', tot_datapath_list)

    if len(tot_datapath_list) <= 1:
        print('CANNOT RUN WITH 0, 1 DATA SET')
        raise FileNotFoundError

    shuffle(tot_datapath_list)

    fin_datapath_list = []
    # path : /home/bjh/obsk/v_nas2/I66/DICOM/train/abnormal/I66_mri_AB_00000039

    for path in tot_datapath_list:
        datasaver = LabelDataSaver.DataSaver(path)
        datasaver.saveYdata_labeled()
        x_path = path + '/img/x'
        y_path = path + '/img/y'
        if os.path.isdir(y_path) is True:
            if len(os.listdir(y_path)) == len(os.listdir(x_path)):
                fin_datapath_list.append(path)

    dataset_cnt = len(fin_datapath_list) # n
    # b_size = 1
    # k = k_fold

    data_loader = TrainDataLoader.DataLoader(data_path_list=fin_datapath_list, k_fold=k_fold, c_size=256, i_channel=1, n_class=2)
    trainer = Trainer(data_loader=data_loader, model_id=model_id, optimizer=optimizer, learning_rate=learning_rate, cost_name=cost, act_func=act_func, layer_n=layer_cnt)
    trainer.train(n_epochs=1, n_t_iters=(math.ceil(dataset_cnt / k_fold * (k_fold - 1))-1) * 8, n_v_iters=math.ceil(dataset_cnt / k_fold) * 8, b_size=1, keep_prob=drop_out_rate)


def tr_main_process():
    # DB 체크 후 process 정보 가져옴
    time.sleep(5)

    # client = MongoClient('mongodb://172.16.52.79:27017')
    # db = client.ohif_deployment
    # cursor = db.training.find()

    cur = OraDB.prepareCursor()
    cur.execute("select tr_status, tr_model_info, layer_cnt, tr_validation, loss_info, optimizer_info, learning_rate, drop_out_rate, activation_info, tr_model_id, normal_data, abnormal_data from training")

    for row in cur:
        tr_status, tr_model_info, layer_cnt, tr_validation, loss_info, optimizer_info, learning_rate, drop_out_rate, activation_info, tr_model_id, normal_data, abnormal_data = list(row)
        if tr_status == 'running':
            try:
                # 학습 시작
                cur.execute("update training set tr_status='running_now' where tr_model_id=:tr_model_id", {'tr_model_id':tr_model_id})
                OraDB.dbCommit()
                print('Train Started')
                train_start_time = time.time()

                # print(row)
                train_run(tr_model_info=tr_model_info, layer_cnt=layer_cnt, tr_validation=tr_validation,
                          loss_info=loss_info, optimizer_info=optimizer_info, learning_rate=learning_rate, drop_out_rate=drop_out_rate,
                          act_func=activation_info, tr_model_id=tr_model_id, normal_data=normal_data, abnormal_data=abnormal_data)

                # 학습 종료되면 tr_status를 end로 갱신
                print('Training Finished')
                train_end_time = time.time()
                duration = (train_end_time - train_start_time) // 60
                cur.execute("update training set tr_status='end', duration=:duration where tr_model_id=:tr_model_id", {'duration':str(duration), 'tr_model_id':tr_model_id})
                OraDB.dbCommit()
                OraDB.releaseConn()

            except:
                # 에러 발생시 tr_status를 fail로 갱신
                cur.execute("update training set tr_status='fail' where tr_model_id=:tr_model_id", {'tr_model_id':tr_model_id})
                OraDB.dbCommit()
                OraDB.releaseConn()
                print('Error Occurred')


# process runner
if __name__ == "__main__":
    tr_main_process()
    time.sleep(1)
    print('train process finished')


