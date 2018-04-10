"""
Default Model Creator, Made by BJH. OBS Korea
"""

from Oracle_connector import OraDB

OraDB.createConn(OraDB.INFO)
cur = OraDB.prepareCursor()

default_model_create_query = "INSERT INTO ADMIN.TRAINING VALUES(:tr_model_id, :tr_model_name, :tr_validation, :tr_status, :stop_yn, " \
                             ":tr_progress, :st_runtime, :ed_runtime, :normal_data, :abnormal_data, :file_path, :tr_model_info, " \
                             ":layer_cnt, :activation_info, :loss_info, :optimizer_info, :learning_rate, :drop_out_rate, " \
                             ":duration, :epoch_num, :epoch_accuracy, :epoch_cost, :ruserid)"

query_values = [2, 'DEFAULT_MODEL', 20, 'end', 'n', 100, None, None, None, None, None, None, 13, 'prelu', 'dsc', 'adam', 1e-4, 0.3, None, 50, 100, None, 1]

cur.execute(default_model_create_query, query_values)

OraDB.dbCommit()

OraDB.releaseConn()

