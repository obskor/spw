# -*- coding: utf-8 -*-

"""
Main Processor Module, Made by BJH, CGM, JYJ OBS Korea
"""


import subprocess
import time
from Oracle_connector import OraDB
from collections import deque


OraDB.createConn(OraDB.INFO)


if __name__ == "__main__":
    while True:
        time.sleep(5)
        cur = OraDB.prepareCursor()

        ts_queue = deque()

        cur.execute("SELECT dl_test_id, dl_status FROM dl_test")

        for row in cur:
            dl_test_id, dl_status = list(row)
            if dl_status == 'run':
                ts_queue.append({'dl_test_id': dl_test_id})

        OraDB.releaseConn()

        for i in range(len(ts_queue)):
                test_proc = subprocess.Popen(['python', 'test_process.py'])
                test_proc.wait()
