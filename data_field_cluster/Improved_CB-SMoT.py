# scoding:utf-8
import math
import numpy as np
import Utility
import matplotlib.pylab as pl
from collections import defaultdict
import show_track


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

# 代码跑不通，改进型的CB-SMoT也没明显特征
def caculate_mul_sigma(points):
    distance_list = []
    total_len = len(points)
    for i in range(0, total_len-1):
        distance_list.append(Utility.distance_calculate(points[i], points[i+1]))
    sum = 0
    # distance_threshold = caculate_distance_threshold(distance_list)
    distance_list.sort()
    pl.plot(distance_list, 'g')
    pl.show()
    count = 0
    for i in range(len(distance_list)):
        if distance_list[i] < distance_threshold:
            sum += distance_list[i]
            count += 1
    result = []             #第一个元素存均值mul，第二个元素存标准方差sigma
    mul = sum/count
    result.append(mul)
    sum = 0
    for i in range(len(distance_list)):
        if distance_list[i] < distance_threshold:
            ele = distance_list[i] - mul
            sum += math.pow(ele, 2)
    sigma = math.sqrt(sum / count)
    result.append(sigma)
    return result


def caculate_distance_threshold(distance_list):
    distance_list.sort()
    # 画平滑前的密度序列图（未归一化）
    pl.figure(figsize=Utility.figure_size)
    l = pl.plot(distance_list, 'og')
    pl.setp(l, markersize=3)
    pl.xlabel('sequence')
    pl.ylabel('density')
    pl.show()
    max_potential = max(distance_list)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_potential_list = Utility.calculate_conv(distance_list, kernel)
    # 画平滑后的密度序列图
    pl.figure(figsize=Utility.figure_size)
    l2 = pl.plot(smoothed_potential_list, 'og')
    pl.setp(l2, markersize=3)
    pl.xlabel('sequence')
    pl.ylabel('density')
    pl.show()
    curvature_index = Utility.calculate_curvature3(smoothed_potential_list, True)

    if curvature_index > 0:
        threshold_candid = smoothed_potential_list[curvature_index]
        if max_potential / threshold_candid > 2:
            print '*' * 30
            print max_potential
            print threshold_candid
            print '*' * 30
            threshold_candid = max_potential / math.sqrt(2)
        threshold = threshold_candid
        return threshold
    else:
        return -1


def calculate_dist_threshold(dist_list):
    dist_list.sort(reverse=True)
    # max_dist = max(dist_list)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_dist_list = Utility.calculate_c

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
def cb_smot(points, min_time, area):
    eps = caculate_eps(points, area)
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
    core_point_idx = [pointIdx for pointIdx, surPointIdxs
                      in surround_points.iteritems() if
                      (points[max(surPointIdxs)].time - points[min(surPointIdxs)].time) >= min_time]

    # 邻域内包含某个核心点的非核心点，定义为边界点
    border_point_idx = []
    for point_idx, sur_points_idx in surround_points.iteritems():
        if point_idx not in core_point_idx:
            for idx in sur_points_idx:
                if idx in core_point_idx:
                    if point_idx not in border_point_idx:
                        border_point_idx.append(point_idx)
                    break
                    # 噪音点既不是边界点也不是核心点
        noise_point_idx = [point_idx for point_idx in range(len(points)) if
                           point_idx not in core_point_idx and point_idx not in border_point_idx]

        groups = [idx for idx in range(len(points))]
        # 各个核心点与其邻域内的所有核心点放在同一个簇中
        for pointidx, surroundIdxs in surround_points.iteritems():
            for oneSurroundIdx in surroundIdxs:
                if pointidx in core_point_idx and oneSurroundIdx in core_point_idx and pointidx < oneSurroundIdx:
                    for idx in range(len(groups)):
                        if groups[idx] == groups[oneSurroundIdx]:
                            groups[idx] = groups[pointidx]
        # 边界点跟其邻域内的某个核心点放在同一个簇中
        for pointidx, surroundIdxs in surround_points.iteritems():
            for oneSurroundIdx in surroundIdxs:
                if pointidx in border_point_idx and oneSurroundIdx in core_point_idx:
                    groups[pointidx] = groups[oneSurroundIdx]
                    break
        # 获取所有的分组
        clusters = defaultdict(list)
        for i, group_idx in enumerate(groups):
            if i not in noise_point_idx:
                clusters[group_idx].append(points[i])
        return clusters


def main(file_name):
    show_track.show_track(file_name)
    # data = Utility.read_data_file(file_name)  # 注意修改数据文件时要修改读文件程序
    data = Utility.read_geolife_data_file(file_name)
    compressed_data = Utility.data_preprocessing_no_compress(data)
    clusters = cb_smot(compressed_data, minTime, area)
    print "total clusters num=" + str(len(clusters))
    for id, points in clusters.iteritems():
        lon = [point.lon for point in points]
        lat = [point.lat for point in points]
        pl.plot(lon, lat, 'or')
    pl.show()






f_name = u"D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_022\geolife_022_2009-07-09.txt"
# f_name = 'data\IMIS_3_DAY_152.txt'
minTime = 5 * 60 * 1000
area = 0.3

main(f_name)
