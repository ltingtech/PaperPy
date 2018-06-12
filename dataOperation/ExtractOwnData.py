# scoding:utf-8
import time
import datetime
import csv
import os
import os.path
import db_connect as db_mysql


# 解析自己记录的CSV数据，并将其拼成SQL语句进行插入数据库中
def process(data_dir, temp_file_name):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    traid_begin = 20095
    pointid_begin = 119443809
    dataid = 500
    objid = 500001
    UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%fz'
    now_stamp = time.time()
    local_time = datetime.datetime.fromtimestamp(now_stamp)
    utc_time = datetime.datetime.utcfromtimestamp(now_stamp)
    offset = local_time - utc_time
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            temp_file = open(temp_file_name, 'w')
            csvfile = file(data_dir + '\\' + file_name, 'r')
            reader = csv.reader(csvfile)
            line_number = 1
            traname = '\'' + file_name.split('.')[0] + '\''
            is_first = True
            count = 0
            for line in reader:
                if line_number <= 4:
                    line_number += 1
                    continue
                else:
                    count += 1
                    longitude = float(line[3])
                    latitude = float(line[2])
                    height = round(float(line[4]))
                    utc_time = line[8]
                    local_time = datetime.datetime.strptime(utc_time, UTC_FORMAT) + offset
                    t = '\'' + local_time.strftime('%Y-%m-%d %H:%M:%S') + '\''
                    if is_first:
                        from_time = t
                        is_first = False
                    to_time = t
                    insert_sql = "INSERT INTO owndata_point(pointid, traid, longitude, latitude,time_date,height) VALUES" \
                                 " (%d,%d,%f,%f,%s,%d);" % (pointid_begin, traid_begin, longitude, latitude, t, height)
                    temp_file.write(insert_sql + '\n')
                    pointid_begin += 1
            temp_file.close()
            insert_trajectory_sql = "INSERT INTO owndata_trajectory(traid,traname,dataid,objid,starttime,endtime,totalpoint)" \
                                    " VALUES (%d,%s,%d,%d,%s,%s,%d);" % \
                                    (traid_begin, traname, dataid, objid, from_time, to_time, count)
            traid_begin += 1
            csvfile.close()
            print insert_trajectory_sql
            db = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                        database='stdatamining', charset='utf8')
            cur = db.cursor()
            cur.execute(insert_trajectory_sql)
            db.commit()
            f_r = open(temp_file_name, 'r')
            for line_sql in f_r.readlines():
                sql = line_sql.split('\n')[0]
                cur.execute(sql)
                db.commit()
            cur.close()
            f_r.close()




    # csvfile = file('E:\Science\data\\11_10.csv', 'r')
    #
    # reader = csv.reader(csvfile)
    # line_number = 1
    # traid_begin = 20000
    # pointid_begin = 119208807
    # dataid = 500
    # objid = 500001
    # traname = '\'' + '11_10' + '\''
    # UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%fz'
    # now_stamp = time.time()
    # local_time = datetime.datetime.fromtimestamp(now_stamp)
    # utc_time = datetime.datetime.utcfromtimestamp(now_stamp)
    # offset = local_time - utc_time
    # is_first = True
    # count = 0
    # for line in reader:
    #     if line_number <= 4:
    #         line_number += 1
    #         continue
    #     else:
    #         count += 1
    #         longitude = float(line[3])
    #         latitude = float(line[2])
    #         height = round(float(line[4]))
    #         utc_time = line[8]
    #         local_time = datetime.datetime.strptime(utc_time, UTC_FORMAT) + offset
    #         t = '\'' + local_time.strftime('%Y-%m-%d %H:%M:%S') + '\''
    #         if is_first:
    #             from_time = t
    #             is_first = False
    #         to_time = t
    #         insert_sql = "INSERT INTO geolife_point(pointid, traid, longitude, latitude,time_date,height) VALUES" \
    #                      " (%d,%d,%f,%f,%s,%d);" % (pointid_begin, traid_begin, longitude, latitude, t, height)
    #         temp_file.write(insert_sql + '\n')
    # temp_file.close()
    # insert_trajectory_sql = "INSERT INTO geolife_trajectory(traid,traname,dataid,objid,starttime,endtime,totalpoint)" \
    #                           " VALUES (%d,%s,%d,%d,%s,%s,%d);" % \
    #                         (traid_begin, traname, dataid, objid, from_time, to_time, count)
    # print insert_trajectory_sql
    # csvfile.close()


process(u"E:\Science\data\OwnData\\test", u"E:\Science\data\\temp.txt")
