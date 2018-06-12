# _*_coding:utf-8_*_# 
import datetime
import time
import csv


def convert(txt_file):
    # obj_id,  traj_id,  subtraj_id,  time,  lon,  lat,   port_id, date
    f = open(txt_file, 'r')
    i = 0
    is_title = True
    is_first_line = True
    last_ymd = ''
    csv_writer = csv.writer(open('dd.csv', 'wb'))
    while True :
        i += 1
        point = f.readline()
        # line += 1
        if is_title:
            is_title = False
            continue
        if point:
            record = [point.split(', ')[3], point.split(', ')[7], point.split(', ')[5],
                      point.split(', ')[0], point.split(', ')[1], point.split(', ')[2],
                      point.split(', ')[8], point.split(', ')[0]]
            ymd = record[3].split(' ')[0]
            t = datetime.datetime.strptime(record[3], "%Y-%m-%d %H:%M:%S")
            d = t.timetuple()
            ss = str(int(time.mktime(d)*1000))  # 毫秒
            record[3] = ss
            '''
            data = [[point.split(', ')[3], point.split(', ')[7], point.split(', ')[5], 
            point.split(', ')[0], point.split(', ')[1], point.split(', ')[2], 
            point.split(', ')[8]] for point in open(txt_file, 'r')]
            '''
            if last_ymd == ymd: 
                change_file = False
            else:
                # fileRecord = 0
                change_file = True
                is_first_line = True
                last_ymd = ymd
            '''
            if line > lineLimit:
                change_file = True
                is_first_line = True
                last_ymd = ymd
            '''
            if change_file:
                # 初始化化
                line = 0
                csvfile_name = ymd+'.csv'
                print csvfile_name
                # csvfile_name = ymd+'_'+str(fileRecord)+'.csv'
                # fileRecord += 1
                change_file = False
                csv_writer = csv.writer(open(csvfile_name, 'wb'))
            if is_first_line:
                csv_writer.writerow(['obj_id', 'traj_id', 'subtraj_id', 'time', 'lon', 'lat', 'port_id', 'date'])
                is_first_line = False
            else:   
                csv_writer.writerow(record)
        else:
            f.close()
            break

convert("imis1month.txt")
