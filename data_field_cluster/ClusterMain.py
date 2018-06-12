# _*_coding:utf-8_*_

import data_field_cluster as dfc
import numpy as np
import show_track
import Utility
import matplotlib.pyplot as plt
import pylab as pl

import time


def main(file_name):
    start_time = time.time()
    show_track.show_track(file_name)
    data = Utility.read_data_file(file_name)   # 注意修改数据文件时要修改读文件程序
    # data = Utility.read_geolife_data_file(file_name)
    compressed_data = Utility.data_preprocessing(data)
    '''
    # 绘制压缩后的轨迹点
    pl.figure(figsize=Utility.figure_size)
    x = [point.lon for point in compressed_data]
    y = [point.lat for point in compressed_data]
    l1 = pl.plot(x, y, 'og')
    pl.setp(l1, markersize=3)
    pl.xlabel("longitude")
    pl.ylabel("latitude")
    pl.show()
    '''

    num_limit = 51
    dist_2_array = Utility.calculate_dist_2_array(compressed_data)
    dist_2_array_copy = []
    for arr in dist_2_array:  #  备份起来，最后的优化处理会用到
        arr_list = []
        for ele in arr:
            arr_list.append(ele)
        dist_2_array_copy.append(arr_list)
    partial_dist_list = Utility.calculate_partial_dist(dist_2_array,  num_limit)
    Utility.normalize_dist_array(dist_2_array,  partial_dist_list)
    max_partial_dist = max(partial_dist_list)
    optimal_sigma = 0.3
    velocity_sigma = 0.5
    '''
    print '*' * 20 + 'the optimal sigma is '
    print optimal_sigma
    print 'the max partial dist is '
    print max_partial_dist
    '''

    dfc.normalize_point_velocity(compressed_data)  # 速度归一化
    velocity_list = [point.velocity for point in compressed_data]
    potential_list = Utility.calculate_potential_value(dist_2_array,  velocity_list,
                                                       optimal_sigma, velocity_sigma, num_limit)
    # 备份
    temp_potential_list = []
    for i in range(len(potential_list)):
        temp_potential_list.append(potential_list[i])

    '''
     # 画未归一化的密度序列图（未排序）
    xx = [float(i)/len(compressed_data) for i in range(len(compressed_data))]
    pl.figure(figsize=Utility.figure_size)
    pl.plot(xx,  potential_list, linewidth=Utility.line_width)
    pl.xlabel('x')
    pl.ylabel('density')
    pl.show()
    '''

    #  画出potential的统计图  采用备份数据，函数里面需要排序
    potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)

    #  更新距离矩阵，加入时间距离
    dfc.refresh_dist_array(compressed_data,  dist_2_array)
    # 计算每个聚类点的距离特征；
    density_dist_list = dfc.calculate_density_distance(dist_2_array,  potential_list)  #  画出distance图
    '''
    xx = range(len(compressed_data))
    pl.plot(xx,  density_dist_list)
    pl.show()
    '''

    dfc.add_attributes(compressed_data,  potential_list,  density_dist_list)  #  完善了数据的potential和dist 属性

    temp_distance = []
    for i in range(len(density_dist_list)):
        temp_distance.append(density_dist_list[i])
    dist_threshold = dfc.calculate_dist_threshold(temp_distance)

    # ************直接画出聚类2016.10.27*******************************
    cluster_points = dfc.get_cluster_points(compressed_data, potential_threshold)
    cluster_lon = [point.lon for point in cluster_points]
    cluster_lat = [point.lat for point in cluster_points]

    # 获得所有的聚类
    cluster_list = []
    is_next_cluster = False
    cluster = []
    for i in range(len(cluster_points)-1):
        if i == (len(cluster_points)-4):
            print 'why'
        cluster.append(cluster_points[i])
        # 如果需要分成下一个聚类，则要重新定义一个元组
        if is_next_cluster:
            cluster_list.append(cluster)
            cluster = []
        distance = Utility.distance_calculate(cluster_points[i], cluster_points[i+1])
        time_interval = cluster_points[i+1].time - cluster_points[i].time
        # 聚类之间的距离阈值为200米，时间间隔阈值为30分钟
        if distance > 800 or time_interval > 30*60*1000:
            is_next_cluster = True
        else:
            is_next_cluster = False
    cluster_list.append(cluster)




    # comp_file = open(u"D:\Python\PaperPy\DataOperation\compress.txt", 'w')
    # for point in compressed_data:
    #     comp_file.write(str(point.lon) + ' ' + str(point.lat) + ' ' + str(point.time) + '\n')
    # comp_file.close()
    # cluster_file = open(u"D:\Python\PaperPy\DataOperation\cluster.txt", 'w')
    # for point in cluster_points:
    #     cluster_file.write(str(point.lon) + ' ' + str(point.lat) + ' ' + str(point.time) + '\n')
    # cluster_file.close()

    # 绘制压缩后的轨迹点
    pl.figure(figsize=Utility.figure_size)
    # x = [point.lon for point in compressed_data]
    # y = [point.lat for point in compressed_data]
    # l1 = pl.plot(x, y, 'og')
    # pl.setp(l1, markersize=3)
    pl.xlabel("longitude")
    pl.ylabel("latitude")
    colors = ['or', 'ok', 'ob']
    for i in range(len(cluster_list)):
        lon = []
        lat = []
        for j in range(len(cluster_list[i])):
            lon.append(cluster_list[i][j].lon)
            lat.append(cluster_list[i][j].lat)
        pl.plot(lon, lat, colors[i % len(colors)])

    pl.show()


    # temp_distance = []
    # for i in range(len(density_dist_list)):
    #     temp_distance.append(density_dist_list[i])
    # dist_threshold = dfc.calculate_dist_threshold(temp_distance)
    # '''
    # print 'potential threshold'
    # print potential_threshold
    # print 'distance threshold'
    # print dist_threshold
    # '''
    #
    #
    # centre_potential = []
    # centre_dist = []
    # centre_index_list = []  #  直接存放聚类中心点
    # if potential_threshold > 0 and dist_threshold > 0:
    #     for i in range(len(density_dist_list)):
    #         if potential_list[i] > potential_threshold and density_dist_list[i] > dist_threshold:
    #             centre_potential.append(potential_list[i])
    #             centre_dist.append(density_dist_list[i])
    #             centre_index_list.append(i)
    #
    # else:
    #     print 'there are something wrong with the threshold'
    #
    # '''
    # #  画出potential_distance图
    # #  结果显示
    # pl.plot(potential_list,  density_dist_list,  'ob')
    # pl.plot(centre_potential, centre_dist, 'oy')
    # pl.xlabel('density')
    # pl.ylabel('distance')
    # pl.show()
    # '''
    #
    # stop_point_list = dfc.get_stop_position(compressed_data,  centre_index_list)
    # dfc.result_show(data,  stop_point_list)
    #
    # '''
    # for point in stop_point_list:
    #     print str(point.lon) + ', ' + str(point.lat) + ', ' + \
    #           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(point.time / float(1000)))
    # '''
    #
    # print '*' * 20 + 'after merge....'
    # centre_index_merge = dfc.merge_stop_position(dist_2_array_copy,  centre_index_list, potential_list)
    # new_centre_index_list = dfc.refine_stop_position(dist_2_array_copy, centre_index_merge, compressed_data)
    # new_stop_point_list = dfc.get_stop_position(compressed_data,  new_centre_index_list)
    #
    # # *******************************
    # # 画经过提纯处理后的decision graph
    # new_centre_potential = []
    # new_centre_dist = []
    # for index in new_centre_index_list:
    #     new_centre_potential.append(potential_list[index])
    #     new_centre_dist.append(density_dist_list[index])
    #
    # '''
    # plt.figure(figsize=Utility.figure_size)
    # pl.plot(potential_list, density_dist_list, 'ob')
    # pl.plot(new_centre_potential, new_centre_dist, 'oy')
    # pl.xlabel('density')
    # pl.ylabel('distance')
    # pl.show()
    # '''
    # dfc.result_show(data,  new_stop_point_list)
    #
    # '''
    # for point in new_stop_point_list:
    #     print str(point.lon) + ', ' + str(point.lat) + ', ' + \
    #           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(point.time / float(1000)))
    # '''
    #


# file_name = u"D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_022\geolife_022_2009-07-16.txt"
file_name = 'data\IMIS_3_DAY_152.txt'
# file_name = 'data\\5_28.txt'
main(file_name)
