# -*- coding: utf-8 -*-

"""
Main Processor Module, Made by BJH, CGM, JYJ OBS Korea
"""


import subprocess
import time
from Oracle_connector import OraDB

OraDB.createConn(OraDB.INFO)


if __name__ == "__main__":
    while True:
        time.sleep(5)
        cur = OraDB.prepareCursor()
        cur.execute("select tr_model_id, tr_status, stop_yn from training")
        # training
        for row in cur:
            tr_model_id, tr_status, stop_yn = list(row)
            if tr_status == 'running':
                train_proc = subprocess.Popen(['python', 'train_process.py'])
                train_proc.wait()
        OraDB.releaseConn()




