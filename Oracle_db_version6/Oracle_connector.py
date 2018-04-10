"""
Default Model Creator, Made by BJH. OBS Korea
"""

import cx_Oracle


class OraDB:
    INFO = 'HIRA01/HIRA01@172.16.52.79:1521/hira'
    ADMIN = None

    @classmethod
    def createConn(cls,info):
        OraDB.ADMIN = cx_Oracle.connect(info)
        print('Database Connector Created')

    @classmethod
    def prepareCursor(cls):
        print('Database Cursor Created')
        return OraDB.ADMIN.cursor()

    @classmethod
    def dbCommit(cls):
        return OraDB.ADMIN.commit()

    @classmethod
    def releaseConn(cls):
        OraDB.prepareCursor().close()

