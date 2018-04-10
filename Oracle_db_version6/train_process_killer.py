# -*- coding: utf-8 -*-

"""
Train Processor Killer Module, Made by BJH, CGM, JYJ OBS Korea
"""

import os
import time
from Oracle_connector import OraDB

OraDB.createConn(OraDB.INFO)


while True:
    time.sleep(5)
    cur = OraDB.prepareCursor()
    cur.execute("select tr_model_id, tr_status, stop_yn from training")

    for row in cur:
        tr_model_id, tr_status, stop_yn = list(row)

        if tr_status == 'cancel' and stop_yn == 'Y':
            os.system('. ./train_process_killer.sh')

            cur.execute("update training set stop_yn='N' where tr_model_id=:tr_model_id", [tr_model_id])
            OraDB.dbCommit()
    OraDB.releaseConn()

    time.sleep(5)            

