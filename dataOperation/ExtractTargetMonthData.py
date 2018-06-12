# _*_coding:utf-8_*_# 
import datetime
import time


def extract_1month_data(id, txt_file):
    f = open(txt_file, 'r')
    wf = open('month_data_2262.txt', 'wb')
    while True:
        point = f.readline()
        if point:
            record = [point.split(', ')[3], point.split(', ')[7], point.split(', ')[5],
                      point.split(', ')[0], point.split(', ')[1], point.split(', ')[2],
                      point.split(', ')[8], point.split(', ')[0]]
            if record[0] == id:
                ymd = record[3].split(' ')[0]
                t = datetime.datetime.strptime(record[3], "%Y-%m-%d %H:%M:%S")
                d = t.timetuple()
                ss = str(int(time.mktime(d)*1000))  # 毫秒
                record[3] = ss
                content_str = ', '.join(record)
                wf.write(content_str)
                wf.write('\n')
            else:
                continue
        else:
            f.close()
            wf.close()
            break


def extract_3month_data(id, txt_file, file_name):
    f = open(txt_file, 'r')
    wf = open(file_name, 'wb')
    count = 0
    while True:
        count += 1
        point = f.readline()
        if point:
            record = [point.split(' ')[1], point.split(' ')[2], point.split(' ')[3].split('\n')[0],
                      point.split(' ')[0]+'000']
            if record[0] == id:
                content_str = ', '.join(record)
                wf.write(content_str)
                wf.write('\n')
            else:
                continue
        else:
            f.close()
            wf.close()
            break

# extract_1month_data('2262', "imis1month.txt")
extract_3month_data('2281', "E:\\Science\\data\\Imis3month\\data\\original.txt",
                    "data\\imis_3_month\\IMIS_3_month_2281_copy.txt")
