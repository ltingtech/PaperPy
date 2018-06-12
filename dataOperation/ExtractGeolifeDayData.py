# _*_coding:utf-8_*_# 

import datetime
import read_db_data
import db_connect as db_mysql
import os



# oracle
def extract_data():
    # 起始日期
    start_time = '2008-10-01 00:00:00'
    # 结束时间
    end_time = '2008-10-01 23:59:59'
    while start_time < '2010-01-01 00:00:00':
        file_name = 'data\\geolife_data\\geolife_014_'
        file_name += start_time.split(' ')[0]+'.txt'
        sql_head = "select longitude, latitude, time_date from GEOLIFE_POINT_014 t "
        sql_condition = "where  t.time_date < '"
        sql_condition += end_time
        sql_condition += "' and t.time_date > '"
        sql_condition += start_time
        sql_condition += "' order by time_date"
        record_count = read_db_data.get_record_count('select count(*) from GEOLIFE_POINT_014 t '+sql_condition)
        print record_count
        if record_count > 0:
            sql = sql_head+sql_condition
            print sql
            db = read_db_data.connection()
            cur = db.cursor()
            cur.execute(sql)
            f = open(file_name, 'w')
            for result in cur:
                line = ''
                for ele in result:
                    line += str(ele)
                    line += ','
                f.write(line+'\n')
            f.close()
            cur.close()
            
        # 日期按天自增
        t1 = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        start_time = str(t1+datetime.timedelta(days=1))
        end_time = str(t2+datetime.timedelta(days=1))


# mysql
def extract_data_mysql():
    # 起始日期
    start_time = '2008-10-01 00:00:00'
    # 结束时间
    end_time = '2008-10-01 23:59:59'
    while start_time < '2010-01-01 00:00:00':
        file_name = 'data\\geolife_data\\geolife_153_'
        file_name += start_time.split(' ')[0]+'.txt'
        sql_head = "select longitude, latitude, time_date from GEOLIFE_153 t "
        sql_condition = "where  t.time_date < '"
        sql_condition += end_time
        sql_condition += "' and t.time_date > '"
        sql_condition += start_time
        sql_condition += "' order by time_date"
        record_count = db_mysql.get_record_count('select count(*) from GEOLIFE_153 t '+sql_condition)
        print record_count
        if record_count > 0:
            sql = sql_head+sql_condition
            print sql
            connection = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                                database='stdatamining', charset='utf8')
            cursor = connection.cursor()
            try:
                cursor.execute(sql)
                results = cursor.fetchall()
                f = open(file_name, 'w')
                for record in results:
                    line = ''
                    for ele in record:
                        line += str(ele)
                        line += ','
                    f.write(line+'\n')
            except Exception:
                print 'Wrong'
            f.close()
            cursor.close()
            connection.close()

        # 日期按天自增
        t1 = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        start_time = str(t1+datetime.timedelta(days=1))
        end_time = str(t2+datetime.timedelta(days=1))



# mysql
# 批量选取出数据，抽取出的轨迹应该是时间连续的 ，时间长度大于8小时
def extract_batch_data():
    db = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                                database='stdatamining', charset='utf8')
    cur = db.cursor()
    traId_list_sql = "SELECT objid,traid FROM continuous_8h_track"
    print traId_list_sql
    cur.execute(traId_list_sql)
    result = cur.fetchall()
    traId_list = [[ele for ele in record] for record in result]  #选出所有满足条件的objId和traId
    cur.close()
    # 先创建目录
    if not os.path.exists('data\\geolife_data\\geolife_data_8h'):
        os.makedirs('data\\geolife_data\\geolife_data_8h')
    for objId_traId in traId_list:
        file_name = 'data\\geolife_data\\geolife_data_8h\\'
        objId = objId_traId[0]
        traId = objId_traId[1]
        txt_name = str(objId) + '_' + str(traId) + '.txt'
        file_name += txt_name
        select_sql = 'SELECT longitude, latitude, time_date FROM geolife_point  WHERE traid=' + \
                     str(traId) + ' ORDER BY time_date'
        print select_sql
        cur = db.cursor()
        cur.execute(select_sql)
        result = cur.fetchall()
        f = open(file_name, 'w')
        for record in result:
            line = ''
            for ele in record:
                line += str(ele)
                line += ','
            f.write(line + '\n')
        f.close()
        cur.close()
    db.close()







# extract_data_mysql()
extract_batch_data()
