# scoding:utf-8
import pylab as pl
import data_field_cluster as dfc
import numpy as np
import show_track
import Utility
from TrackPoint import TrackPoint
from collections import defaultdict
import os
import os.path
import time


# def apply_dbscan2(points, dist_array, eps, min_pts):
#     surround_points = defaultdict(list)
#     # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
#     for i in range(len(points)):
#         for j in range(i+1, len(points)):
#             if dist_array[i][j-i] <= eps:
#                 surround_points[i].append(j)
#                 surround_points[j].append(i)
#
#     # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
#     core_point_idx = [pointIdx for pointIdx, surPointIdxs
#                       in surround_points.iteritems() if len(surPointIdxs) >= min_pts]
#
#     # 邻域内包含某个核心点的非核心点，定义为边界点
#     border_point_idx = []
#     for point_idx, sur_points_idx in surround_points.iteritems():
#         if point_idx not in core_point_idx:
#             for idx in sur_points_idx:
#                 if idx in core_point_idx:
#                     if point_idx not in border_point_idx:
#                         border_point_idx.append(point_idx)
#                     break
#     # 噪音点既不是边界点也不是核心点
#     noise_point_idx = [point_idx for point_idx in range(len(points)) if
#                        point_idx not in core_point_idx and point_idx not in border_point_idx]
#
#     groups = [idx for idx in range(len(points))]
#     # 各个核心点与其邻域内的所有核心点放在同一个簇中
#     for pointidx, surroundIdxs in surround_points.iteritems():
#         for oneSurroundIdx in surroundIdxs:
#             if pointidx in core_point_idx and oneSurroundIdx in core_point_idx and pointidx < oneSurroundIdx:
#                 for idx in range(len(groups)):
#                     if groups[idx] == groups[oneSurroundIdx]:
#                         groups[idx] = groups[pointidx]
#     # 边界点跟其邻域内的某个核心点放在同一个簇中
#     for pointidx, surroundIdxs in surround_points.iteritems():
#         for oneSurroundIdx in surroundIdxs:
#             if pointidx in border_point_idx and oneSurroundIdx in core_point_idx:
#                 groups[pointidx] = groups[oneSurroundIdx]
#                 break
#     # 获取所有的分组
#     clusters = defaultdict(list)
#     for i, group_idx in enumerate(groups):
#         if i not in noise_point_idx:
#             clusters[group_idx].append(points[i])
#     return clusters


def apply_dbscan(points, dist_array, eps, min_pts):
    surround_points = defaultdict(list)
    # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if dist_array[i][j-i] <= eps:
                surround_points[i].append(j)
                surround_points[j].append(i)

    # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
    for point_idx, surPointIdxs in surround_points.iteritems():
        if len(surPointIdxs) > min_pts:
            points[point_idx].is_core_point = True
    for point_idx, point in enumerate(points):
        if not point.is_core_point:
            for idx in surround_points[point_idx]:
                if points[idx].is_core_point:
                    point.is_border_point = True
                    break
    for point in points:
        if not point.is_core_point and not point.is_border_point:
            point.is_noise_point = True

    groups = [idx for idx in range(len(points))]
    for pointidx, surroundIdxs in surround_points.iteritems():
        if points[pointidx].is_core_point:
            for oneSurroundIdx in surroundIdxs:
                if pointidx < oneSurroundIdx and points[oneSurroundIdx].is_core_point:
                    groups[oneSurroundIdx] = groups[pointidx]
        if points[pointidx].is_border_point:
            distinct_group_list = set()
            for oneSurroundIdx in surroundIdxs:
                if points[oneSurroundIdx].is_core_point:
                    distinct_group_list.add(groups[oneSurroundIdx])
            if len(distinct_group_list) == 1:
                groups[pointidx] = min(distinct_group_list)
            else:
                min_group = min(distinct_group_list)
                groups[pointidx] = min_group
                for i in range(len(groups)):
                    if groups[i] in distinct_group_list:
                        groups[i] = min_group
    # 获取所有的分组
    clusters = defaultdict(list)
    for i, group_idx in enumerate(groups):
        if not points[i].is_noise_point:
            clusters[group_idx].append(points[i])
    return clusters


# 调整DBSCAN算法的参数，批量处理geolife数据，把结果数据和图片保存下来
def parameter_group_experiment(data_dir, txt_save_dir, png_save_dir):
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    epss = [50]
    min_ptss = [100]
    for eps in epss:
        for min_pts in min_ptss:
            wf_name = txt_save_dir + '\\' + str(eps) + '_' + str(min_pts) + '.txt'
            f_w = open(wf_name, 'w')
            head_line = 'eps=' + str(eps) + '; minPts=' + str(min_pts) + '\n'
            f_w.write(head_line)
            for parent, dirName, file_names in os.walk(data_dir):
                for file_name in file_names:
                    print file_name
                    f_name = data_dir + '\\' + file_name
                    f_info = 'file_name: ' + file_name + '\n'
                    f_w.write(f_info)
                    # data = Utility.read_geolife_data_file(f_name)
                    data = Utility.read_own_data_file(f_name)
                    data.sort(key=lambda p:p.time)
                    compressed_data = Utility.data_preprocessing(data)
                    dist_2_array = Utility.calculate_dist_2_array(compressed_data)
                    clusters_dic = apply_dbscan(compressed_data, dist_2_array, eps, min_pts)
                    clusters = []
                    for id, points in clusters_dic.iteritems():
                        if len(points) > 0:
                            clusters.append(points)
                    clusters.sort(key=lambda pp: pp[0].time)
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
                    png_file = png_save_dir + '\\' + file_name + '_' + 'nap_' + \
                               str(eps) + 'distSigma_' + str(min_pts) + '.png'
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
    min_pts = 100
    eps = 50
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            print file_name + '\n'
            begin_time = time.time()  # 记录开始时间
            content = file_name + ','
            data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
            compressed_data = Utility.data_preprocessing(data)
            dist_2_array = Utility.calculate_dist_2_array(compressed_data)
            clusters_dic = apply_dbscan(compressed_data, dist_2_array, eps, min_pts)
            clusters = []
            for id, points in clusters_dic.iteritems():
                if len(points) > 0:
                    clusters.append(points)
            clusters.sort(key=lambda p: p[0].time)
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
            png_file = png_save_dir + '\\' + file_name + '_' + 'eps_' + \
                       str(eps) + 'minpts_' + str(min_pts) + '.png'
            pl.savefig(png_file)
            pl.close()
            end_time = time.time()
            time_gap = end_time - begin_time
            point_number = len(compressed_data)
            content += str(point_number) + ','
            content += str(time_gap) + ',\n'
            f_w.write(content)
    f_w.close()


def main(file_name):
    show_track.show_track(file_name)
    start_time = time.time()
    # data = Utility.read_data_file(file_name)   # 注意修改数据文件时要修改读文件程序
    data = Utility.read_geolife_data_file(file_name)
    # data = Utility.read_own_data_file(file_name)
    compressed_data = Utility.data_preprocessing(data)
    dist_2_array = Utility.calculate_dist_2_array(compressed_data)
    clusters = apply_dbscan(compressed_data, dist_2_array, 80, 100)
    end_time = time.time()
    time_gap = end_time - start_time
    print 'consumed time is ' + str(time_gap) + ' seconds'
    for id, points in clusters.iteritems():
        lon = [point.lon for point in points]
        lat = [point.lat for point in points]
        pl.plot(lon, lat, 'or')
    pl.show()

# file_name = "D:\\Python\\PaperPy\\DataOperation\\data\geolife_data\\geolife_022\\geolife_022_2009-01-20.txt"
# file_name = "D:\\Python\\PaperPy\\DataOperation\\data\\own_data\\11_26_am_12_50.txt"
# file_name = 'data\IMIS_3_DAY_152.txt'
# file_name = "D:\\Python\\PaperPy\\track1.txt"
# file_name = "D:\\Python\\PaperPy\\track1.txt"

# main(file_name)

# parameter_group_experiment("D:\\Python\\PaperPy\\data_filed_cluster\\experience_result\\new experiment\\"
#                            "NewDBSCAN\\compare_experiment\\DBSCAN\\geolife_subtrajectory2",
#                            "D:\\Python\\PaperPy\\data_filed_cluster\\experience_result\\new experiment\\"
#                            "NewDBSCAN\\compare_experiment\\DBSCAN\\geolife_data_txt_result2",
#                            "D:\\Python\\PaperPy\\data_filed_cluster\\experience_result\\new experiment\\NewDBSCAN\\"
#                            "compare_experiment\\DBSCAN\\geolife_data_png_result2")

parameter_group_experiment(
    "D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\DBSCAN\\own_data_experiment\\own_data",
"D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\DBSCAN\\own_data_experiment\\own_data_txt_result",
"D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\DBSCAN\\own_data_experiment\\own_data_png_result")

data_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
           u'\compare_experiment\DBSCAN\efficient_compare\data'
txt_save_file = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
                u'\compare_experiment\DBSCAN\efficient_compare\\result.txt'
png_save_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
               u'\compare_experiment\DBSCAN\efficient_compare\png_result'

# compare_efficiency(data_dir, txt_save_file, png_save_dir)