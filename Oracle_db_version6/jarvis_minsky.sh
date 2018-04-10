#!/bin/bash

#VENV=/home/obsk/Javis_dl_system

WORKDIR=/home/admin/DL_system/Oracle_db_version

DAEMON1=test_main_process.py
DAEMON2=train_main_process.py
DAEMON3=train_process_killer.py

#LOG1=/var/log/javis_test.log
#LOG2=/var/log/javis_train.log

LOG1=${WORKDIR}/log/javis_test.log
LOG2=${WORKDIR}/log/javis_train.log
LOG3=${WORKDIR}/log/javis_cancel.log

function do_start()
{
	source activate ohif
        cd ${WORKDIR}
        nohup python3.5 ${DAEMON1} >> ${LOG1} & nohup python3.5 ${DAEMON2} >> ${LOG2} & nohup python3.5 ${DAEMON3} >> ${LOG3} &
}

function do_stop()
{

        PID=`ps -ef | grep ${DAEMON1} | grep -v grep | awk '{print $2}'`
        if [ "$PID" != "" ]; then
                kill -9 $PID
        fi

        PID=`ps -ef | grep ${DAEMON2} | grep -v grep | awk '{print $2}'`
        if [ "$PID" != "" ]; then
                kill -9 $PID
	    fi

        PID=`ps -ef | grep ${DAEMON3} | grep -v grep | awk '{print $2}'`
        if [ "$PID" != "" ]; then
                kill -9 $PID
        fi
}

case "$1" in
    start|stop)
        do_${1}
        ;;
    reload|restart)
        do_stop
        do_start
        ;;
    *)
        echo "Usage: /etc/init.d/tunnel {start|stop|restart}"
        exit 1
        ;;
esac
