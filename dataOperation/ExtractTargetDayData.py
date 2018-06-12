# _*_coding:utf-8_*_# 

import os
import read_db_data
import show_track


# 根据目标号提取一段时间内每天的数据
def extract_data(obj_id):
    db = read_db_data.connection()
    data_file = r"D:\Python\PaperPy\DataOperation\data\IMIS_"+obj_id
    png_file = r"D:\Python\PaperPy\DataOperation\png\IMIS_"+obj_id
    os.makedirs(data_file)
    os.makedirs(png_file)
    table_id = 1
    sql_model = "select obj_id, traj_id, time, lon, lat from %s  t where t.obj_id = '%s' order by t.time"
    while table_id < 21:
        table_name = 'IMIS_7_'
        table_name += str(table_id)
        sql = sql_model % (table_name, obj_id)
        file_name = data_file+'\\'
        file_name += table_name+'_'+obj_id+".txt"
        read_db_data.select_sql2(db, sql, file_name)
        table_id += 1
    
    show_track.show_track_in_dir(data_file, png_file)
        
    
extract_data('981')
