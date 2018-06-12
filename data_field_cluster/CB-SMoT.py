# scoding:utf-8
import math
import numpy as np
import Utility
import matplotlib.pylab as pl
from collections import defaultdict
import show_track
import os
import os.path
import re
import time


def caculate_ck(num):
    ck = []
    ck.append(1)
    begin_idx = 1
    while begin_idx < num:
        ele = 0
        for m in range(begin_idx):
            ele += ck[m] * ck[begin_idx-1-m] * 1.0 / ((m+1) * (2*m+1))
        ck.append(ele)
        begin_idx += 1
    return ck


def erf(ck, x):
    total_len = len(ck)
    result = 0
    for k in range(total_len):
        result += ck[k]/(2*k+1) * math.pow((math.sqrt(math.pi)/2*x), (2*k+1))
    return result


def F_function(p, mul, sigma, ck):
    result = mul + sigma * math.sqrt(2) * erf(ck, 2*p-1)
    return result


def caculate_mul_sigma(points):
    distance_list = []
    total_len = len(points)
    for i in range(0, total_len-1):
        distance_list.append(Utility.distance_calculate(points[i], points[i+1]))
    sum = 0
    for i in range(len(distance_list)):
        print distance_list[i]

    for i in range(len(distance_list)):
        sum += distance_list[i]
    pl.plot(distance_list, 'g')
    pl.show()
    result = []             #第一个元素存均值mul，第二个元素存标准方差sigma
    mul = sum/len(distance_list)
    result.append(mul)
    sum = 0
    for i in range(len(distance_list)):
        if distance_list[i] < mul*5:
            ele = distance_list[i] - mul
            sum += math.pow(ele, 2)
    sigma = math.sqrt(sum / len(distance_list))
    result.append(sigma)
    return result


def caculate_eps(points, area):
    # data = Utility.read_geolife_data_file(file_name)
    # compressed_data = Utility.data_preprocessing(data)
    # pl.show()
    mul_sigma = caculate_mul_sigma(points)
    mul = mul_sigma[0]
    sigma = mul_sigma[1]
    print 'mul=' + str(mul)
    print 'sigma=' + str(sigma)
    ck = caculate_ck(100)
    eps = F_function(area, mul, sigma, ck)
    print 'eps=' + str(eps)
    return eps


# DBSCAN的变种
# def cb_smot(points, min_time, eps):
#     # eps = caculate_eps(points, area)
#     # eps = 100
#     surround_points = defaultdict(list)
#     # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
#     for i in range(len(points)):
#         sum_distance = 0
#         idx = i-1
#         surround_points[i].append(i)
#         while idx >= 0 and sum_distance <= eps:
#             surround_points[i].append(idx)
#             sum_distance += Utility.distance_calculate(points[idx], points[i])
#             idx -= 1
#         sum_distance = 0
#         idx = i+1
#         while idx < len(points) and sum_distance <= eps:
#             surround_points[i].append(idx)
#             sum_distance += Utility.distance_calculate(points[idx], points[i])
#             idx += 1
#     # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
#     core_point_idx = [pointIdx for pointIdx, surPointIdxs
#                       in surround_points.iteritems() if
#                       (points[max(surPointIdxs)].time - points[min(surPointIdxs)].time) >= min_time]
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

def cb_smot(points, min_time, eps):
    # eps = caculate_eps(points, area)
    # eps = 100
    surround_points = defaultdict(list)
    # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
    for i in range(len(points)):
        sum_distance = 0
        idx = i-1
        surround_points[i].append(i)
        while idx >= 0 and sum_distance <= eps:
            surround_points[i].append(idx)
            sum_distance += Utility.distance_calculate(points[idx], points[i])
            idx -= 1
        sum_distance = 0
        idx = i+1
        while idx < len(points) and sum_distance <= eps:
            surround_points[i].append(idx)
            sum_distance += Utility.distance_calculate(points[idx], points[i])
            idx += 1
    # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
    for point_idx, surPointIdxs in surround_points.iteritems():
        if (points[max(surPointIdxs)].time - points[min(surPointIdxs)].time) >= min_time:
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
                for i in range(len(groups)):
                    if groups[i] in distinct_group_list:
                        groups[i] = min_group
    # 获取所有的分组
    clusters = defaultdict(list)
    for i, group_idx in enumerate(groups):
        if not points[i].is_noise_point:
            clusters[group_idx].append(points[i])
    return clusters



# 调节CB-SMoT的参数，对实验轨迹进行批量处理，做对比实验，最终结果保存在一个.txt文件中,图片也保存下来
def parameter_group_experiment(data_dir, txt_save_dir, png_save_dir):
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    MinTimes = [10*60*1000]
    epss = [100]
    for m_time in MinTimes:
        for eps in epss:
            t_minute = int(m_time / 1000 / 60)
            wf_name = txt_save_dir + '\\'+str(eps)+'_'+str(t_minute)+'minutes.txt'
            wf_name2 = txt_save_dir + '\\' + str(eps) + '_' + str(t_minute) + 'minutes_cluster_info.txt'
            f_w = open(wf_name, 'w')
            f_w2 = open(wf_name2, 'w')
            head_line = 'eps='+str(eps)+'; mintTime=' + str(t_minute)+'minutes\n'
            f_w.write(head_line)
            for parent, dirName, file_names in os.walk(data_dir):
                for file_name in file_names:
                    f_name = data_dir + '\\' + file_name
                    f_info = 'file_name: ' + file_name + '\n'
                    f_w.write(f_info)
                    data = Utility.read_own_data_file(f_name)
                    compressed_data = Utility.data_preprocessing_no_compress(data)
                    clusters_dic = cb_smot(compressed_data, m_time, eps)
                    clusters = []
                    for id, points in clusters_dic.iteritems():
                        if len(points) > 0:
                            clusters.append(points)
                    clusters.sort(key=lambda p: p[0].time)
                    merged_clusters = clusters
                    # Utility.print_clusters_info(merged_clusters)
                    # Utility.show_clusters(merged_clusters)
                    # Utility.show_clusters_trajectory(clusters, compressed_data)
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
                    pl.title(file_name + '_' + str(len(merged_clusters)) + '_' + 'clusters')
                    png_file = png_save_dir + '\\' + file_name + '_' + 'eps_' +\
                               str(eps) + 'minTim_' + str(t_minute) + '.png'
                    pl.savefig(png_file)
                    pl.close()
                    # ********************************
                    cluster_info = str(len(clusters)) + '\n'
                    f_w.write(cluster_info)
                    f_w2.write(str(len(merged_clusters)) + '\n')
                    for cluster in merged_clusters:
                        f_w2.write(str(len(cluster)) + '\n')
                        cluster.sort(key=lambda pp: pp.time)
                        length = len(cluster)
                        first_line = str(cluster[0].lon) + ',' + str(cluster[0].lat) + ',' + cluster[0].time_str
                        end_line = str(cluster[length - 1].lon) + ',' + str(cluster[length - 1].lat) + \
                                   ',' + cluster[length - 1].time_str + '\n'
                        f_w.write(first_line)
                        f_w.write(end_line)
                        for point in cluster:
                            p_content = str(point.lon) + ',' + str(point.lat) + ',\n'
                            f_w2.write(p_content)
            f_w.close()
            f_w2.close()


# 选用不同轨迹点个数的轨迹，测试算法的运行时间，以此来对比不同算法的运算效率
def compare_efficiency(data_dir, txt_save_file, png_save_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    f_w = open(txt_save_file, 'w')
    minTime = 10 * 60 * 1000
    eps = 100
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            print file_name + '\n'
            begin_time = time.time()  # 记录开始时间
            content = file_name + ','
            data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
            compressed_data = Utility.data_preprocessing(data)
            clusters_dic = cb_smot(compressed_data, minTime, eps)
            clusters = []
            for id, points in clusters_dic.iteritems():
                if len(points) > 0:
                    clusters.append(points)
            clusters.sort(key=lambda p: p[0].time)
            merged_clusters1 = Utility.merge_clusters(clusters)
            merged_clusters = Utility.clusters_refinement(merged_clusters1)
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
            png_file = png_save_dir + '\\' + file_name + '_' + 'minTime_' + \
                       str(minTime) + 'eps_' + str(eps) + '.png'
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
    # show_track.show_track(file_name)
    start_time = time.time()
    # data = Utility.read_data_file(file_name)  # 注意修改数据文件时要修改读文件程序
    # data = Utility.read_geolife_data_file(file_name)
    data = Utility.read_own_data_file(file_name)
    compressed_data = Utility.data_preprocessing_no_compress(data)
    clusters_dic = cb_smot(compressed_data, minTime, eps)
    clusters = []
    for id, points in clusters_dic.iteritems():
        if len(points) > 0:
            clusters.append(points)
    clusters.sort(key=lambda p: p[0].time)
    # merged_clusters = Utility.merge_clusters(clusters)
    merged_clusters = clusters
    end_time = time.time()
    time_gap = end_time - start_time
    print 'consumed time is ' + str(time_gap) + ' seconds'
    Utility.print_clusters_info(merged_clusters)
    Utility.show_clusters(merged_clusters)
    Utility.show_clusters_trajectory(clusters, compressed_data)







# f_name = u"D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_022\geolife_022_2009-07-11.txt"
# f_name = "D:\\Python\\PaperPy\\DataOperation\\data\\geolife_data\\observe\\1359.txt"
# f_name = "D:\\Python\\PaperPy\\DataOperation\\data\\own_data\\11_26_am_12_50.txt"
# #
# minTime = 5 * 60 * 1000
# # area = 0.3
# eps = 100
# #
# main(f_name)

# data_dir = u"D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN" \
#                u"\compare_experiment\CB-SMotT\own_data_experiment\own_data"
# txt_save_dir = u"D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN" \
#                u"\compare_experiment\CB-SMotT\own_data_experiment\own_data_txt_result"
# png_save_dir = u"D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN" \
#                u"\compare_experiment\CB-SMotT\own_data_experiment\own_data_png_result"
# parameter_group_experiment(data_dir, txt_save_dir, png_save_dir)


data_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
           u'\compare_experiment\CB-SMotT\efficient_compare\data'
txt_save_file = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
                u'\compare_experiment\CB-SMotT\efficient_compare\\result.txt'
png_save_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
               u'\compare_experiment\CB-SMotT\efficient_compare\png_result'

compare_efficiency(data_dir, txt_save_file, png_save_dir)
