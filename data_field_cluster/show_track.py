#  _*_coding:utf-8_*_
import matplotlib.pyplot as plt
import pylab as pl
import Utility


def track_split():    # 根据时间把轨迹分段
    # points = [[float(point.split(', ')[0]), float(point.split(', ')[1]), long(point.split(', ')[2])]
    # for point in open('cruise_track.txt', 'r')]
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
        if i>100:
            break
    file_w.close()
    f.close()


def show_track_in_time(file_name):
    time_format = '%Y/%m/%d %H:%M:%S'
    fig = plt.figure()
    ax = fig.add_subplot(111,  projection='3d')
    '''
    points = [[float(point.split(', ')[0]), float(point.split(', ')[1]), 
             time.mktime(time.strptime(re.split(', |\n', point)[2], time_format))] for point in open(file_name, 'r')]
    '''
    points = [[float(point.split(', ')[3]), float(point.split(', ')[4]), long(point.split(', ')[2])] for point in open(file_name, 'r')]
    
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
    pl.figure(figsize=Utility.figure_size)
    l = pl.plot(x, y, 'og')
    pl.setp(l, markersize=3)
    pl.title('track')
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.show()


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
        
            
# track_split()
# show_track_in_time('splited_track\split_track3.txt')
'''
show_calculate('distcalculate.txt', 'No_Interval')
show_calculate('intervalDist.txt', 'Interval')
show_calculate('directionList.txt', 'Direction')
show_calculate('directionInterval.txt', 'Interval_Direct')
show_calculate('speedList.txt', 'Speed')
show_calculate('speedInterval.txt', 'Speed_Interval')

show_calculate2('distance', 'distcalculate.txt', 'intervalDist.txt')
show_track('track1.txt')

show_calculate3()


show_track('D:\Python\PaperPy\data_filed_cluster\data\IMIS_3_DAY_200.txt')

'''





