# scoding:utf-8
import os
import matplotlib.pyplot as pl


# trucks数据集的解析，画出每条轨迹
def analysis(data_file, png_save_dir, txt_save_dir):
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)
    f = open(data_file, 'r')
    last_tra = 'first'
    lon = []
    lat = []
    tim = []
    for line in f:
        line_ele = line.split(';')
        next_str = line_ele[0] + '_' + line_ele[1]
        if last_tra == 'first':
            last_tra = next_str
            lon.append(line_ele[4])
            lat.append(line_ele[5])
            tim.append(date_formate_convert(line_ele[2], line_ele[3]))
        else:
            if next_str == last_tra:
                lon.append(line_ele[4])
                lat.append(line_ele[5])
                tim.append(date_formate_convert(line_ele[2], line_ele[3]))
            else:
                png_file = png_save_dir + '\\'+last_tra + '.png'
                pl.subplot(121)
                pl.plot(lon, lat, 'g')
                pl.subplot(122)
                pl.plot(lon, lat, 'og')
                pl.savefig(png_file)
                pl.close()
                txt_file = txt_save_dir + '\\' + last_tra + '.txt'
                t_f = open(txt_file, 'w')
                for idx, longitude in enumerate(lon):
                    line = longitude + ',' + lat[idx] + ',' + tim[idx] + ',\n'
                    t_f.write(line)
                t_f.close()
                lon = []
                lat = []
                tim = []
                lon.append(line_ele[4])
                lat.append(line_ele[5])
                tim.append(date_formate_convert(line_ele[2], line_ele[3]))
                last_tra = next_str
    f.close()



# 日期字符串格式转换
def date_formate_convert(dmy, hms):
    d = dmy.split('/')[0]
    m = dmy.split('/')[1]
    y = dmy.split('/')[2]
    new_ymd = y + '-' + m + '-' + d
    t = new_ymd + ' ' + hms
    return t

data_file = "E:\\Science\\data\\trucks\\trucks.txt"
png_save_dir = "E:\\Science\\data\\trucks\\png"
txt_save_dir = "E:\\Science\\data\\trucks\\data"
analysis(data_file, png_save_dir, txt_save_dir)

