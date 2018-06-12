# _*_coding:utf-8_*_# 
import matplotlib.pyplot as plt
import pylab as pl
import read_db_data
import os
import os.path
import db_connect as db_mysql


def track_split():    # 根据时间把轨迹分段
    # points = [[float(point.split(', ')[0]), float(point.split(', ')[1]), long(point.split(', ')[2])]
    #  for point in open('cruise_track.txt', 'r')]
    f = open('cruise_track.txt', 'r')
    line = f.readline()
    last_time = long(line.split(', ')[2])
    # last_time = int(line.split(' ')[2])
    i = 1
    file_name = 'split_track_child1.txt'
    file_w = open(file_name, 'w')
    is_open = True
    while line:
        now_time = long(line.split(', ')[2])
        interval = long(now_time-last_time)
        # now_time = int(line.split(' ')[2])
        # interval = int(now_time-last_time)
        if abs(interval) > 100000000:
            i += 1
            file_w.close
            file_name = 'split_track'+str(i)+'.txt'
            file_w = open(file_name, 'w')
            file_w.write(line)
            last_time = now_time
        else:
            file_w.write(line)
        line = f.readline()
        if i > 100:
            break
    file_w.close()
    f.close()


def show_track_in_time(file_name):
    time_format = '%Y/%m/%d %H:%M:%S'
    fig = plt.figure()
    ax = fig.add_subplot(111,  projection='3d')
    '''
    points = [[float(point.split(', ')[0]), float(point.split(', ')[1]), 
             time.mktime(time.strptime(re.split(', |\n', point)[2], timeFormat))] for point in open(file_name, 'r')]
    '''
    points = [[float(point.split(', ')[0]), float(point.split(', ')[1]), long(point.split(', ')[2])]
              for point in open(file_name, 'og')]
    
    for point in points:
            ax.scatter(point[0], point[1], long(point[2]))
    ax.set_xlabel('Lon Label')
    ax.set_ylabel('Lat Label')
    ax.set_zlabel('Time Label')
    plt.show()


def show_track(file_name):
    
    points = [[float(point.split(',')[1]), float(point.split(',')[0])] for point in open(file_name, 'r')]
    print '打印轨迹点数量：'
    print len(points)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    pl.plot(x, y, 'og')
    pl.title('track')
    
    pl.show()


def show_track2(file_name, save_fig_name):
    points = [[float(point.split(', ')[4]), float(point.split(', ')[5])] for point in open(file_name, 'r')]
    print '打印轨迹点数量：'
    print len(points)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    pl.subplot(121)
    pl.plot(x, y, 'g')
    pl.title('track')
    pl.subplot(122)
    pl.plot(x, y, 'og')
    pl.title('track')
    pl.savefig(save_fig_name)
    pl.close()


def show_calculate2(title_name, *file_names):
    colors = ['r', 'b', 'g', 'y']
    length = len(colors)
    color = 0
    for f in file_names:
        dist_list = [dist for dist in open(f, 'r')]
        index = []
        dist2 = []
        for i, dist in enumerate(dist_list):
            
            index.append(i)
            dist2.append(float(dist))
        pl.plot(index, dist2, colors[int(color%length)])
        color += 1
        
    pl.title(title_name)
    pl.text('距离对比图')
    pl.show()


def show_calculate3():
    dist_list = [dist for dist in open('speedList.txt', 'r')]
    index = []
    dist2 = []
    for i, dist in enumerate(dist_list):
        index.append(i)
        dist2.append(float(dist))
    plot1,  = pl.plot(index, dist2, 'b')    # matplotlib 返回的是一个列表，加个逗号之后就把matplotlib对象从列表中取出了

    dist_list2 = [dist for dist in open('speedInterval.txt', 'r')]
    index2 = []
    dist22 = []
    for i, dist in enumerate(dist_list2):
        index2.append(i)
        dist22.append(float(dist))
    plot2,  = pl.plot(index2, dist22, 'g')
    pl.legend((plot1, plot2), ('speed_no_interval', 'speed_interval'), loc = 'best', numpoints = 1)
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.title('distance analysis')
    pl.show()


def show_calculate(file_names, title_name):       
    dist_list = [dist for dist in open('intervalDist.txt', 'r')]
    index = []
    dist2 = []
    for i, dist in enumerate(dist_list):
        index.append(i)
        dist2.append(float(dist))
    pl.plot(index, dist2, 'g+')
    
    pl.title('distance')
    pl.show()


# 显示目录下所有轨迹数据文件的轨迹
def show_track_in_dir(file_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for parent, dirName, file_names in os.walk(file_dir):
        print '*'*10
        print file_names
        for file_name in file_names:
            print file_name
            f = file_dir+"\\" + file_name
            points = [[float(point.split(',')[1]), float(point.split(',')[0])] for point in open(f, 'r')]
            print '打印轨迹点数量：'
            print len(points)
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            pl.subplot(121)
            pl.plot(x, y, 'g')
            pl.title(file_name.split('.')[0])
            pl.subplot(122)
            pl.plot(x, y, 'og')
            pl.title(file_name.split('.')[0])
            save_fig_name = save_dir+'\\'+file_name.split('.')[0]+'.png'
            pl.savefig(save_fig_name)
            pl.close()

# 显示数据库中的IMIS_3_DAY的数据
def show_db_track(obj_id):
    db = read_db_data.connection()
    sql = "select lon,lat,time from IMIS_3_DAY t where t.obj_id = "+str(obj_id)+" order by time"
    file_name = "data\imis_3_day\IMIS_3_DAY_"+str(obj_id)+".txt"
    read_db_data.select_sql2(db, sql, file_name)
    show_track(file_name)


# 根据traid直接显示数据库中Geolife的轨迹
def show_mysql_geolife_track(traId, file_name):
    db = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                database='stdatamining', charset='utf8')
    cur = db.cursor()
    select_sql = 'SELECT longitude, latitude, time_date FROM geolife_point  WHERE traid=' + \
                 str(traId) + ' ORDER BY time_date'

    lon = []
    lat = []
    try:
        cur.execute(select_sql)
        result = cur.fetchall()
        for record in result:
            lon.append(record[0])
            lat.append(record[1])
        if file_name != 'default':
            file_name += '\\' + str(traId) + '.txt'
            f = open(file_name, 'w')
            for result in cur:
                line = ''
                for ele in result:
                    line += str(ele)
                    line += ','
                f.write(line + '\n')
            f.close()
        cur.close()
    except:
        print 'errot to execute the sql'
    db.close()
    pl.plot(lon, lat, 'g')
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.title('traId_' + str(traId) + '_' + str(len(lon)))
    pl.show()


# 根据文件中存放的traid集从数据库中批量导出数据文件
def extract_data(traid_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    db = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                database='stdatamining', charset='utf8')
    cur = db.cursor()
    f_r = open(traid_file, 'r')
    lines = f_r.readlines()
    for line in lines:
        traid = line.split(',')[0]
        select_sql = 'SELECT longitude, latitude, time_date FROM geolife_point  WHERE traid=' + \
                     str(traid) + ' ORDER BY time_date'
        cur.execute(select_sql)
        result = cur.fetchall()
        data_file = data_dir + '\\' + str(traid) + '.txt'
        f_w = open(data_file, 'w')
        for record in result:
            line = ''
            for ele in record:
                line += str(ele)
                line += ','
            f_w.write(line + '\n')
        f_w.close()
    cur.close()
    db.close


# 完善dataobjrelation数据库表,
def process_dataobjrelation():
    db = db_mysql.db_connection(host_name='localhost', user_name='root', password='root',
                                database='stdatamining', charset='utf8')
    cur = db.cursor()
    select_sql1 = 'SELECT DISTINCT objid FROM geolife_trajectory WHERE objid>400076'
    cur.execute(select_sql1)
    result = cur.fetchall()
    begin = 10501
    for record in result:
        objid = record[0]
        objname = "'" + str(objid)[3:] + "'"
        insert_sql = "insert into dataobjrelation values(%d,400,'geolife',%d,%s);" % (begin, objid, objname)
        print insert_sql
        begin += 1
        cur2 = db.cursor()
        cur2.execute(insert_sql)
        db.commit()
        cur2.close()
    cur.close()
    db.close()







def main():
    db = read_db_data.connection()
    sql = 'select count(*),obj_id from IMIS_7_14 t group by obj_id'
    cur = db.cursor()
    cur.execute(sql)
    obj_id = []
    for ele in cur:
        num = int(ele[0])
        if num > 3000:
            obj_id.append(ele[1])
    cur.close()
    for id in obj_id:
        sql = "select * from IMIS_7_14 t where t.obj_id = "+id+" order by time"
        # save_fig_name1 = "IMIS_1month_process\\IMIS_7_1_figure\\IMIS_7_1_"+id+".png"
        save_fig_name = "data\\chaoqian\\figure\\IMIS_7_14_"+id+"_point.png"
        file_name = "data\\chaoqian\\data\\"+"IMIS_7_4_"+id+".txt"
        read_db_data.select_sql2(db, sql, file_name)
        show_track2(file_name, save_fig_name)

# main()

# show_db_track(130)
# show_track_in_dir("D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment"
#                   "\NewDBSCAN\compare_experiment\CB-SMotT\own_data_experiment\own_data",
#                   "D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment"
#                   "\NewDBSCAN\compare_experiment\CB-SMotT\own_data_experiment\own_data_track")


# track_file = "D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment" \
#              "\NewDBSCAN\compare_experiment\CB-SMotT\own_data_experiment\own_data\\11_22_am_8_20.txt"
#
# show_track(track_file)

traid_file = u'C:\\Users\\Administrator\\Desktop\\t\\t.txt'
data_dir = u'C:\\Users\\Administrator\\Desktop\\t'
extract_data(traid_file, data_dir)


