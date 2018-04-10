#!/bin/bash

#VENV=/home/obsk/Javis_dl_system

WORKDIR=/home/user01/deep01/
#WORKDIR=/home/obsk/Javis_dl_system

DAEMON=train_process.py

#LOG1=/var/log/javis_test.log
#LOG2=/var/log/javis_train.log

#LOG1=${WORKDIR}/log/javis_test.log
#LOG2=${WORKDIR}/log/javis_train.log


cd ${WORKDIR}       

PID=`ps -ef | grep ${DAEMON} | grep -v grep | awk '{print $2}'`

if [ "$PID" != "" ]; then
    kill -9 $PID
fi



