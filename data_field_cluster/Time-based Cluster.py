# scoding:utf-8

import pylab as pl
import data_field_cluster as dfc
import numpy as np
import show_track
import Utility
from TrackPoint import TrackPoint
import os
import os.path
import time


def apply_time_based_cluster(points, time_threshold, cluster_distance, tolerated_distance):
    result_clusters = []
    current_cluster = []
    prev_cluster = []
    start_point = points[0]     # 用于记录每个聚类的开始的那个点
    start_time = 0              # 用于记录每个聚类的开始的点的时间
    end_time = 0
    points_len = len(points)    # 轨迹点的个数
    for i in range(points_len):
        if len(current_cluster) == 0:
            start_point = points[i]
            start_time = points[i].time    # 起始点和其实时间的更新都是只有在聚类开始的地方会执行
            current_cluster.append(start_point)
            continue
        else:
            current_point = points[i]
            distance = Utility.distance_calculate(start_point, current_point)
            if distance <= cluster_distance:
                current_cluster.append(current_point)
                end_time = points[i].time
            else:
                duration = (end_time - start_time) / 1000 / 60   # 停留点的停留时间，单位为minute
                if duration > time_threshold:
                    result_clusters.append(current_cluster)
                    current_cluster = []
                    prev_cluster = []
                else:
                    prev_cluster_len = len(prev_cluster)
                    if prev_cluster_len > 0:
                        time_interval = (start_time - prev_cluster[prev_cluster_len - 1].time) / 1000 / 60
                        dist_beween_clusters = Utility.distance_calculate(current_cluster[0],
                                                                          prev_cluster[prev_cluster_len - 1])
                        if time_interval > time_threshold and dist_beween_clusters < tolerated_distance:
                            current_cluster += prev_cluster
                            current_cluster.sort(key=lambda p:p.time)
                            result_clusters.append(current_cluster)  #加入新聚类
                            prev_cluster = []
                            current_cluster = []
                        else:
                            prev_cluster = current_cluster
                            current_cluster = []
                    else:
                        prev_cluster = current_cluster
                        current_cluster = []
    if ((end_time - start_time) / 1000 / 60) >= time_threshold:
        result_clusters.append(current_cluster)
    return result_clusters


# 调整Time-based Cluster算法的参数，批量处理geolife数据，把结果数据和图片保存下来
def parameter_group_experiment(data_dir, txt_save_dir, png_save_dir):
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    time_thresholds = [6]
    cluster_distances = [80]
    tolerated_distance = 200
    for time_threshold in time_thresholds:
        for cluster_distance in cluster_distances:
            wf_name = txt_save_dir + '\\' + str(time_threshold) + '_' + str(cluster_distance) + '.txt'
            f_w = open(wf_name, 'w')
            head_line = 't_thrsd=' + str(time_threshold) + '; cluster dist=' + str(cluster_distance) + '\n'
            f_w.write(head_line)
            for parent, dirName, file_names in os.walk(data_dir):
                for file_name in file_names:
                    print file_name
                    f_name = data_dir + '\\' + file_name
                    f_info = 'file_name: ' + file_name + '\n'
                    f_w.write(f_info)
                    # data = Utility.read_geolife_data_file(f_name)
                    data = Utility.read_own_data_file(f_name)
                    compressed_data = Utility.data_preprocessing(data)
                    clusters = apply_time_based_cluster(compressed_data, time_threshold, cluster_distance,
                                                        tolerated_distance)
                    # ***************保存图片*****************
                    x = [point.lon for point in compressed_data]
                    y = [point.lat for point in compressed_data]
                    pl.plot(x, y, 'g')
                    colors = ['or', 'ob', 'og', 'oy', 'ok', 'oc']
                    color_idx = 0
                    for points in clusters:
                        lon = [point.lon for point in points]
                        lat = [point.lat for point in points]
                        pl.plot(lon, lat, colors[color_idx % len(colors)])
                        color_idx += 1
                    pl.title(file_name + '_' + str(len(clusters)) + '_' + 'clusters')
                    png_file = png_save_dir + '\\' + file_name + '_' + 't_thrsd' + \
                               str(time_threshold) + 'cluster_dist' + str(cluster_distance) + '.png'
                    pl.savefig(png_file)
                    pl.close()
                    # ********************************
                    cluster_info = str(len(clusters)) + '\n'
                    f_w.write(cluster_info)
                    for cluster in clusters:
                        time_list = [point.time for point in cluster]
                        sum_lon = 0.0
                        sum_lat = 0.0
                        cluster_len = len(cluster)
                        for point in cluster:
                            sum_lon += point.lon
                            sum_lat += point.lat
                        centre_lon = sum_lon / cluster_len
                        centre_lat = sum_lat / cluster_len
                        line = str(centre_lon) + ',' + str(centre_lat) + ',' \
                               + str(min(time_list)) + ',' + str(max(time_list)) + '\n'
                        f_w.write(line)
            f_w.close()


# 选用不同轨迹点个数的轨迹，测试算法的运行时间，以此来对比不同算法的运算效率
def compare_efficiency(data_dir, txt_save_file, png_save_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    f_w = open(txt_save_file, 'w')
    time_threshold = 6
    cluster_distance = 80
    tolerated_distance = 200
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            print file_name + '\n'
            begin_time = time.time()  # 记录开始时间
            content = file_name + ','
            data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
            compressed_data = Utility.data_preprocessing(data)
            clusters = apply_time_based_cluster(compressed_data, time_threshold, cluster_distance,tolerated_distance)
            merged_clusters = clusters
            # ***************保存图片*****************
            x = [point.lon for point in compressed_data]
            y = [point.lat for point in compressed_data]
            pl.plot(x, y, 'g')
            colors = ['or', 'ob', 'og', 'oy', 'ok', 'oc']
            color_idx = 0
            for points in merged_clusters:
                lon = [point.lon for point in points]
                lat = [point.lat for point in points]
                pl.plot(lon, lat, colors[color_idx % len(colors)])
                color_idx += 1
            pl.title(file_name + '_' + str(len(merged_clusters)) + '_' + 'clusters')
            png_file = png_save_dir + '\\' + file_name + '_' + 't_thrsd_' + \
                       str(time_threshold) + 'c_dist_' + str(cluster_distance) + '.png'
            pl.savefig(png_file)
            pl.close()
            end_time = time.time()
            time_gap = end_time - begin_time
            point_number = len(compressed_data)
            content += str(point_number) + ','
            content += str(time_gap) + ',\n'
            f_w.write(content)
    f_w.close()


# 主函数
def main(file_name):
    t_threshold = 10
    cluster_distance = 60
    tolerated_distance = 200
    data = Utility.read_geolife_data_file(file_name)
    compressed_data = Utility.data_preprocessing(data)
    clusters = apply_time_based_cluster(compressed_data, t_threshold, cluster_distance, tolerated_distance)
    # ***************保存图片*****************
    x = [point.lon for point in compressed_data]
    y = [point.lat for point in compressed_data]
    pl.plot(x, y, 'g')
    colors = ['or', 'ob', 'og', 'oy', 'ok', 'oc']
    color_idx = 0
    for points in clusters:
        lon = [point.lon for point in points]
        lat = [point.lat for point in points]
        pl.plot(lon, lat, colors[color_idx % len(colors)])
        color_idx += 1
    pl.title(file_name + '_' + str(len(clusters)) + '_' + 'clusters')
    pl.show()
    pl.close()



# file_name = u"C:\\Users\\Administrator\\Desktop\\data_5\\4105.txt"
# main(file_name)

# parameter_group_experiment(
#     "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\geolife_subtrajectory2",
#     "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\geolife_data_txt_result2",
#     "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\geolife_data_png_result2")

# parameter_group_experiment(
#     "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\own_data_experiment\\own_data",
# "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\own_data_experiment\\own_data_txt_result",
# "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\own_data_experiment\\own_data_png_result")

data_dir = u'D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\efficient_compare\data'
txt_save_file = u'D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\efficient_compare\\result.txt'
png_save_dir = u'D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\Time-based Cluster\\efficient_compare\\png_result'

compare_efficiency(data_dir, txt_save_file, png_save_dir)
