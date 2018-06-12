# scoding:utf-8
import pylab as pl
import data_field_cluster as dfc
import numpy as np
import show_track
import Utility
from TrackPoint import TrackPoint
from collections import defaultdict
from  mpl_toolkits.mplot3d import axes3d
import os
import os.path
import time
import MySQLdb



# # the improved DBSCAN method using the data field to caculate density
# def dbscan_with_datafield(points, density_thrshold, adjacent_num):
#     surround_points = defaultdict(list)
#     # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
#     for i in range(len(points)):
#         j = max(0, i-adjacent_num/2)
#         k = min(i+adjacent_num/2, len(points)-1)
#         while j <= k:
#             surround_points[i].append(j)
#             j += 1
#     # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
#     core_point_idx = [pointIdx for pointIdx, surPointIdxs
#                       in surround_points.iteritems() if points[pointIdx].potential >= density_thrshold]
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
#                     # 噪音点既不是边界点也不是核心点
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

# the improved DBSCAN method using the data field to caculate density
def dbscan_with_datafield(points, density_thrshold, adjacent_num):
    surround_points = defaultdict(list)
    # 计算每个数据点相邻的数据点，邻域距离上限定义为Eps
    for i in range(len(points)):
        j = max(0, i-adjacent_num/2)
        k = min(i+adjacent_num/2, len(points)-1)
        while j <= k:
            surround_points[i].append(j)
            j += 1
    # 定义邻域内相邻的数据点的个数大于MinPts的为核心点
    # core_point_idx = [pointIdx for pointIdx, surPointIdxs
    #                   in surround_points.iteritems() if points[pointIdx].potential >= density_thrshold]
    for point in points:
        if point.potential > density_thrshold:
            point.is_core_point = True
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
    clusters = defaultdict(list)
    for i, group_idx in enumerate(groups):
        if not points[i].is_noise_point:
            clusters[group_idx].append(points[i])
    return clusters


#  our improved DBSCAN, input the original data, output the clusters
def dbscan_process(data, velocity_sigma, nap, sigma1):
    num_limit = nap
    velocity_sigma = velocity_sigma
    optimal_sigma = sigma1

    compressed_data = Utility.data_preprocessing(data)
    lon = [point.lon for point in compressed_data]
    lat = [point.lat for point in compressed_data]
    # pl.plot(lon, lat, 'og')
    # pl.show()
    # 在计算相邻点之间的距离时，把速度也计算出来
    partial_dist_list = Utility.caculate_adjacent_dist_list(compressed_data, num_limit)
    Utility.normalize_adjacent_dist_list(partial_dist_list)
    Utility.caculate_velocity(compressed_data)
    # ****************新增的对速度进行平滑*****************
    original_velocity_list = [point.velocity for point in compressed_data]
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_velocity_list = Utility.calculate_conv(original_velocity_list, kernel)
    # pl.plot(smoothed_velocity_list)
    # # pl.plot(original_velocity_list)
    # pl.xlabel('sequence number')
    # pl.ylabel('velocity')
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\smoothed velocity.png', dpi=200)
    # pl.show()

    velocity_list = dfc.normalize_velocity(smoothed_velocity_list)

    stability_list = Utility.calculate_stability(compressed_data, num_limit)
    # smoothed_stability_list = Utility.calculate_conv(stability_list, kernel)
    # pl.plot(smoothed_stability_list)
    # pl.xlabel('sequence number')
    # pl.ylabel('move ability')
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\smoothed move_ability.png', dpi=200)
    # pl.show()

    # ************************完毕************************

    # potential_list = Utility.calculate_density_value(partial_dist_list, velocity_list, optimal_sigma, velocity_sigma)
    potential_list = Utility.calculate_density_value(partial_dist_list, stability_list, optimal_sigma, velocity_sigma)

    # ****************新增的对密度进行平滑*****************
    # ************************完毕************************
    # 备份
    temp_potential_list = []
    for i in range(len(potential_list)):
        temp_potential_list.append(potential_list[i])

    # 画出potential的统计图  采用备份数据，函数里面需要排序
    potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)
    density_dist_list = range(len(compressed_data))
    dfc.add_attributes(compressed_data, potential_list, density_dist_list)  # 完善了数据的potential和dist 属性
    clusters_dic = dbscan_with_datafield(compressed_data, potential_threshold, num_limit)  # 返回的clusters 是字典型的
    clusters = []
    for id, points in clusters_dic.iteritems():
        if len(points) > 0:
            clusters.append(points)
    clusters.sort(key=lambda pp: pp[0].time)
    return clusters



# 调整New_DBSCAN算法的参数，批量处理geolife数据，把结果数据和图片保存下来
def parameter_group_experiment(data_dir, txt_save_dir, png_save_dir):
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    naps = [51]
    distance_sigmas = [0.3]
    for nap in naps:
        for dist_sigma in distance_sigmas:
            wf_name = txt_save_dir + '\\' + str(nap) + '_' + str(dist_sigma) + '.txt'
            wf_name2 = txt_save_dir + '\\' + str(nap) + '_' + str(dist_sigma) + '_cluster_info.txt'
            f_w = open(wf_name, 'w')
            f_w2 = open(wf_name2, 'w')
            head_line = 'nap=' + str(nap) + '; distance_sigma=' + str(dist_sigma) + '\n'
            f_w.write(head_line)
            for parent, dirName, file_names in os.walk(data_dir):
                for file_name in file_names:
                    f_name = data_dir + '\\' + file_name
                    f_info = 'file_name: ' + file_name + '\n'
                    f_w.write(f_info)
                    # data = Utility.read_geolife_data_file(f_name)
                    data = Utility.read_own_data_file(f_name)
                    compressed_data = Utility.data_preprocessing(data)
                    partial_dist_list = Utility.caculate_adjacent_dist_list(compressed_data, nap)
                    Utility.normalize_adjacent_dist_list(partial_dist_list)
                    Utility.caculate_velocity(compressed_data)
                    velocity_sigma = 0.5
                    '''
                    # 这些是按速度计算的那些
                    original_velocity_list = [point.velocity for point in compressed_data]
                    interpolated_idx = []  # 保存插值点的索引值
                    for idx, p in enumerate(compressed_data):
                        if p.is_interpolated:
                            interpolated_idx.append(idx)
                    kernel = Utility.generate_gaus_kernel(4)
                    # kernel = [float(1) / 30 for i in range(31)]
                    smoothed_velocity_list = Utility.calculate_conv(original_velocity_list, kernel)

                    velocity_list = dfc.normalize_velocity(smoothed_velocity_list)
                    '''
                    stability_list = Utility.calculate_stability(compressed_data, nap)
                    potential_list = Utility.calculate_density_value(partial_dist_list, stability_list, dist_sigma,
                                                                     velocity_sigma)

                    # ************************完毕************************
                    # 备份
                    temp_potential_list = []
                    for i in range(len(potential_list)):
                        temp_potential_list.append(potential_list[i])
                    #  画出potential的统计图  采用备份数据，函数里面需要排序
                    potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)
                    density_dist_list = range(len(compressed_data))
                    dfc.add_attributes(compressed_data, potential_list, density_dist_list)  # 完善了数据的potential和dist 属性
                    clusters_dic = dbscan_with_datafield(compressed_data, potential_threshold,
                                                         nap)  # 返回的clusters 是字典型的
                    clusters = []
                    for id, points in clusters_dic.iteritems():
                        if len(points) > 0:
                            clusters.append(points)
                    clusters.sort(key=lambda pp: pp[0].time)
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
                    png_file = png_save_dir + '\\' + file_name + '_' + 'nap_' + \
                               str(nap) + 'distSigma_' + str(dist_sigma) + '.png'
                    pl.savefig(png_file)
                    pl.close()
                    # ********************************
                    cluster_info = 'cluster count= ' + str(len(merged_clusters)) + '\n'
                    f_w.write(cluster_info)
                    f_w2.write(str(len(merged_clusters)) + '\n')
                    for cluster in merged_clusters:
                        f_w2.write(str(len(cluster)) + '\n')
                        cluster.sort(key=lambda pp: pp.time)
                        length = len(cluster)
                        first_line = str(cluster[0].lon) + ',' + str(cluster[0].lat) + ',' + cluster[0].time_str
                        end_line = str(cluster[length-1].lon) + ',' + str(cluster[length-1].lat) + \
                                   ',' + cluster[length-1].time_str + '\n'
                        f_w.write(first_line)
                        f_w.write(end_line)
                        for point in cluster:
                            p_content = str(point.lon) + ',' + str(point.lat) + ',\n'
                            f_w2.write(p_content)
            f_w.close()
            f_w2.close()


# 最新的根据停留点数目的变化来观察参数的选择，参数对比实验
def parameter_choose(data_dir, dist_sigmas):
    f_w = open(u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN'
               u'\compare_experiment\\new_DBSCAN\parameter_compare\\parameter_compare.txt', 'w')
    velocity_sigma = 0.5
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            stop_lists = []  #用来暂时存放结果的,根据nap来存放，一个nap值对应一条线
            f_w.write(file_name + '\n')
            naps = [21, 51, 71, 101, 151]
            n = 0
            for num_limit in naps:
                f_w.write('*'*10 + '\n')
                stop_list = []
                for optimal_sigma in dist_sigmas:
                    f_w.write('nap= ' + str(num_limit) + '  dist_sigma=' + str(optimal_sigma) + '\n')
                    print 'n'
                    n += 1
                    data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
                    compressed_data = Utility.data_preprocessing(data)
                    partial_dist_list = Utility.caculate_adjacent_dist_list(compressed_data, num_limit)
                    Utility.normalize_adjacent_dist_list(partial_dist_list)
                    stability_list = Utility.calculate_stability(compressed_data, num_limit)
                    potential_list = Utility.calculate_density_value(partial_dist_list, stability_list, optimal_sigma,
                                                                     velocity_sigma)
                    # 备份
                    temp_potential_list = []
                    for i in range(len(potential_list)):
                        temp_potential_list.append(potential_list[i])
                    #  画出potential的统计图  采用备份数据，函数里面需要排序
                    potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)
                    density_dist_list = range(len(compressed_data))
                    dfc.add_attributes(compressed_data, potential_list,
                                       density_dist_list)  # 完善了数据的potential和dist 属性
                    clusters_dic = dbscan_with_datafield(compressed_data, potential_threshold,
                                                         num_limit)  # 返回的clusters 是字典型的
                    clusters = []
                    for id, points in clusters_dic.iteritems():
                        if len(points) > 0:
                            clusters.append(points)
                    clusters.sort(key=lambda pp: pp[0].time)
                    merged_clusters1 = Utility.merge_clusters(clusters)
                    # merged_clusters1 = clusters  # 不采用合并
                    merged_clusters = Utility.clusters_refinement(merged_clusters1)
                    count = len(merged_clusters)
                    stop_list.append(count)
                    f_w.write('count=' + str(count) + '\n')
                stop_lists.append(stop_list)
            for s_list in stop_lists:
                pl.plot(dist_sigmas, s_list)
                pl.xlabel('dist_sigma')
                pl.xlabel('the number of stops')
            pl.show()
    f_w.close()


# 横坐标是nap，纵坐标是停留点的个数，对nap参数进行选择的实验
def parameter_nap_choose(data_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    velocity_sigma = 0.5
    sigma1_list = [0.2, 0.3, 0.4, 0.5, 0.7]
    nap_list = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151]
    # 每个sigma对应一条曲线
    for sigma1 in sigma1_list:
        f_name = result_dir + '\\' + 'sigma=' + str(sigma1) + '#.txt'
        f_w = open(f_name, 'w')
        stop_number_list = []    #用于记录每个nap对应的的停留点的个数
        for nap in nap_list:    #对每一个(sigma1, nap）组合进行计算停留点的个数
            for parent, dirName, file_names in os.walk(data_dir):
                stops_count = 0
                for file_name in file_names:
                    data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
                    clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
                    # merged_clusters1 = Utility.merge_clusters(clusters)
                    # count = len(merged_clusters1)
                    count = len(clusters)
                    stops_count += count
                content = str(nap) + ',' + str(stops_count) + ',\n'   # 每行记录表示格式为<nap,stops_count,>
                f_w.write(content)
                stop_number_list.append(stops_count)
        pl.plot(nap_list, stop_number_list, 'vg-')
        pl.savefig(result_dir + '\\sima1=' + str(sigma1) + '.png', dpi=400)
        pl.close()
        f_w.close()


# parameter_nap参数选择的配套函数，根据得到的txt结果绘制图片
def parameter_nap_plot(result_dir):
    plot_arr = []
    plot_model = ['^b--', '.r--', '*k--', 'vb:', '*k:']
    sigma_list = []
    max_nap = 0
    plot_count = 0
    min_nap = 1000
    max_stops_num = 0
    min_stops_num = 1000
    for parent, dirName, file_names in os.walk(result_dir):
        for file_name in file_names:
            is_first_line = True
            sigma = file_name.split('=')[1].split('#')[0]
            sigma_list.append(sigma)
            f_r = open(result_dir + '\\' + file_name, 'r')
            nap_list = []
            stop_number_list = []
            for line in f_r.xreadlines():
                if is_first_line:
                    is_first_line = False
                    continue
                nap = float(line.split(',')[0])
                num = float(line.split(',')[1])
                nap_list.append(nap)
                stop_number_list.append(num)
                if nap > max_nap:
                    max_nap = nap
                if nap < min_nap:
                    min_nap = nap
                if num > max_stops_num:
                    max_stops_num = num
                if num < min_stops_num:
                    min_stops_num = num
            p, = pl.plot(nap_list, stop_number_list, plot_model[plot_count])
            plot_count += 1
            plot_arr.append(p)
    pl.xlabel('nap')
    pl.ylabel('number of stops')
    nap_interval = (max_nap - min_nap) / 10
    num_interval = (max_stops_num - min_stops_num) / 10
    pl.xlim(min_nap - nap_interval, max_nap + nap_interval)
    pl.ylim(min_stops_num - num_interval, max_stops_num + num_interval)
    pl.legend((plot_arr[0], plot_arr[1], plot_arr[2], plot_arr[3], plot_arr[4]), (
        r'$\sigma_1=$' + str(sigma_list[0]), r'$\sigma_1=$' + str(sigma_list[1]),
        r'$\sigma_1=$' + str(sigma_list[2]), r'$\sigma_1=$' + str(sigma_list[3]), r'$\sigma_1=$' + str(sigma_list[4])),
              loc='best', numpoints=1)
    pl.savefig(result_dir + '\\result.png', dpi=200)
    pl.show()


# 利用一条轨迹进行计算处理，分析nap参数变化对聚类的SSE系数的影响
def nap_sse_analysis(file_name, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    velocity_sigma = 0.5
    sigma1_list = [0.2, 0.3, 0.4, 0.5, 0.7]
    nap_list = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151]
    # 每个sigma对应一条曲线
    for sigma1 in sigma1_list:
        f_name = result_dir + '\\result\\' + 'sigma=' + str(sigma1) + '#.txt'
        f_w = open(f_name, 'w')
        sse_list = []  # 用于记录每个nap对应的的停留点的个数
        for nap in nap_list:  # 对每一个(sigma1, nap）组合进行计算停留点的个数
            data = Utility.read_geolife_data_file(file_name)
            clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
            sse = Utility.calculate_sse_coeff(clusters)
            content = str(nap) + ',' + str(sse) + ',\n'  # 每行记录表示格式为<nap,stops_count,>
            f_w.write(content)
            sse_list.append(sse)
        pl.plot(nap_list, sse_list, 'vg-')
        pl.savefig(result_dir + '\\png\\sima1=' + str(sigma1) + '.png', dpi=400)
        pl.close()
        f_w.close()

# parameter_nap参数选择的配套函数，根据得到的txt结果绘制图片
def nap_sse_plot(result_dir):
    plot_arr = []
    plot_model = ['^b--', '.r--', '*k--', 'vb:', '*k:']
    sigma_list = []
    plot_count = 0
    max_nap = 0
    min_nap = 1000
    max_sse = 0
    min_sse = 1000
    for parent, dirName, file_names in os.walk(result_dir):
        for file_name in file_names:
            sigma = file_name.split('=')[1].split('#')[0]
            sigma_list.append(sigma)
            f_r = open(result_dir + '\\' + file_name, 'r')
            nap_list = []
            sse_list = []
            for line in f_r.xreadlines():
                nap = float(line.split(',')[0])
                sse = float(line.split(',')[1])
                nap_list.append(nap)
                sse_list.append(sse)
                if nap > max_nap:
                    max_nap = nap
                if nap < min_nap:
                    min_nap = nap
                if sse > max_sse:
                    max_sse = sse
                if sse < min_sse:
                    min_sse = sse
            p, = pl.plot(nap_list, sse_list, plot_model[plot_count])
            plot_count += 1
            plot_arr.append(p)
    pl.xlabel('nap')
    pl.ylabel('sse')
    nap_interval = (max_nap - min_nap) / 100
    sse_interval = (max_sse - min_sse) / 10
    pl.xlim(min_nap - nap_interval, max_nap + nap_interval)
    pl.ylim(min_sse - sse_interval, max_sse + sse_interval)
    pl.legend((plot_arr[0], plot_arr[1], plot_arr[2], plot_arr[3], plot_arr[4]), (
        r'$\sigma_1=$' + str(sigma_list[0]), r'$\sigma_1=$' + str(sigma_list[1]),
        r'$\sigma_1=$' + str(sigma_list[2]), r'$\sigma_1=$' + str(sigma_list[3]),
        r'$\sigma_1=$' + str(sigma_list[4])),
              loc='best', numpoints=1, ncol=1)
    pl.savefig(result_dir + '\\result.png', dpi=200)
    pl.show()


# 横坐标是sigma，纵坐标是停留点的个数，对sigma参数进行选择的实验
def parameter_sigma1_choose(data_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    velocity_sigma = 0.5
    nap_list = [31, 51, 71, 91, 111, 151]
    sigma1_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1]
    # 每个nap对应一条曲线
    for nap in nap_list:
        f_name = result_dir + '\\' + 'nap=' + str(nap) + '#.txt'
        f_w = open(f_name, 'w')
        stop_number_list = []    #用于记录每个nap对应的的停留点的个数
        for sigma1 in sigma1_list:    #对每一个(sigma1, nap）组合进行计算停留点的个数
            for parent, dirName, file_names in os.walk(data_dir):
                stops_count = 0
                for file_name in file_names:
                    data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
                    clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
                    # merged_clusters1 = Utility.merge_clusters(clusters)
                    # count = len(merged_clusters1)
                    count = len(clusters)
                    stops_count += count
                content = str(sigma1) + ',' + str(stops_count) + ',\n'   # 每行记录表示格式为<nap,stops_count,>
                f_w.write(content)
                stop_number_list.append(stops_count)
        pl.plot(sigma1_list, stop_number_list, 'vg-')
        pl.savefig(result_dir + '\\nap=' + str(nap) + '.png', dpi=400)
        pl.close()
        f_w.close()


# parameter_sigma参数选择的配套函数，根据得到的txt结果绘制图片
def parameter_sigma_plot(result_dir):
    plot_arr = []
    plot_model = ['^b--', '.r--', '*k--', 'vb:', '*k:']
    nap_list = []
    plot_count = 0
    max_sigma = 0
    min_sigma = 1000
    max_stops_num = 0
    min_stops_num = 1000
    for parent, dirName, file_names in os.walk(result_dir):
        for file_name in file_names:
            nap = file_name.split('=')[1].split('#')[0]
            nap_list.append(nap)
            f_r = open(result_dir + '\\' + file_name, 'r')
            sigma_list = []
            stop_number_list = []
            for line in f_r.xreadlines():
                sigma = float(line.split(',')[0])
                num = float(line.split(',')[1])
                sigma_list.append(sigma)
                stop_number_list.append(num)
                if sigma > max_sigma:
                    max_sigma = sigma
                if sigma < min_sigma:
                    min_sigma = sigma
                if num > max_stops_num:
                    max_stops_num = num
                if num < min_stops_num:
                    min_stops_num = num
            # sigma_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.7, 0.75, 0.8, 0.9, 1]
            p, = pl.plot(sigma_list, stop_number_list, plot_model[plot_count])
            plot_count += 1
            plot_arr.append(p)
    pl.xlabel(r'$\sigma_1$')
    pl.ylabel('number of stops')
    # max_sigma = 1
    # min_sigma = 0.3
    sigma_interval = (max_sigma - min_sigma) / 10
    num_interval = (max_stops_num - min_stops_num) / 10
    pl.xlim(min_sigma - sigma_interval, max_sigma + sigma_interval)
    # pl.ylim(min_stops_num - num_interval, max_stops_num + num_interval)
    pl.ylim(min_stops_num - num_interval, 50)
    pl.legend((plot_arr[2], plot_arr[4], plot_arr[1], plot_arr[3], plot_arr[0]), (
        r'nap=' + str(nap_list[2]), r'nap' + str(nap_list[4]),
        r'nap' + str(nap_list[1]), r'nap=' + str(nap_list[3]),
        r'nap=' + str(nap_list[0])),
              loc='best', numpoints=1, ncol=2)
    pl.savefig(result_dir + '\\result.png', dpi=200)
    pl.show()


# 利用一条轨迹进行计算处理，分析sigma参数变化对聚类的SSE系数的影响
def sigma_sse_analysis(file_name, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    velocity_sigma = 0.5
    nap_list = [31, 51, 71, 91, 111, 151]
    sigma1_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1]
    # 每个sigma对应一条曲线
    for nap in nap_list:
        f_name = result_dir + '\\result\\' + 'nap=' + str(nap) + '#.txt'
        f_w = open(f_name, 'w')
        sse_list = []  # 用于记录每个nap对应的的停留点的个数
        for sigma1 in sigma1_list:  # 对每一个(sigma1, nap）组合进行计算停留点的个数
            data = Utility.read_geolife_data_file(file_name)
            clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
            sse = Utility.calculate_sse_coeff(clusters)
            content = str(sigma1) + ',' + str(sse) + ',\n'  # 每行记录表示格式为<nap,stops_count,>
            f_w.write(content)
            sse_list.append(sse)
        pl.plot(sigma1_list, sse_list, 'vg-')
        pl.savefig(result_dir + '\\png\\nap=' + str(nap) + '.png', dpi=400)
        pl.close()
        f_w.close()


# parameter_sigma参数选择的配套函数，根据得到的txt结果绘制图片
def sigma_sse_plot(result_dir):
    plot_arr = []
    plot_model = ['^b--', '.r--', '*k--', 'vb:', '*k:']
    nap_list = []
    plot_count = 0
    max_sigma = 0
    min_sigma = 1000
    max_sse = 0
    min_sse = 1000
    for parent, dirName, file_names in os.walk(result_dir):
        for file_name in file_names:
            nap = file_name.split('=')[1].split('#')[0]
            nap_list.append(nap)
            f_r = open(result_dir + '\\' + file_name, 'r')
            sigma_list = []
            sse_list = []
            for line in f_r.xreadlines():
                sigma = float(line.split(',')[0])
                sse = float(line.split(',')[1])
                sigma_list.append(sigma)
                sse_list.append(sse)
                if sigma > max_sigma:
                    max_sigma = sigma
                if sigma < min_sigma:
                    min_sigma = sigma
                if sse > max_sse:
                    max_sse = sse
                if sse < min_sse:
                    min_sse = sse
            p, = pl.plot(sigma_list, sse_list, plot_model[plot_count])
            plot_count += 1
            plot_arr.append(p)
    pl.xlabel(r'$\sigma_1$')
    pl.ylabel('sse')
    sigma_interval = (max_sigma - min_sigma) / 100
    sse_interval = (max_sse - min_sse) / 10
    pl.xlim(min_sigma - sigma_interval, max_sigma + sigma_interval)
    pl.ylim(min_sse - sse_interval, max_sse + sse_interval)
    pl.legend((plot_arr[2], plot_arr[3], plot_arr[4], plot_arr[0], plot_arr[1]), (
        r'nap=' + str(nap_list[2]), r'nap=' + str(nap_list[3]),
        r'nap=' + str(nap_list[4]), r'nap=' + str(nap_list[0]),
        r'nap=' + str(nap_list[1])),
              loc='upper left', numpoints=1, ncol=1)
    pl.savefig(result_dir + '\\result.png', dpi=200)
    pl.show()


# parameter sigma of weight（权重sima）参数的选取
def parameter_sigma_ma_choose(data_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    nap_sigma1_list = [[91, 0.3], [91, 0.5]]
    velocity_sigma_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.8, 0.9, 1]
    # 每个nap对应一条曲线
    for nap_sigma1 in nap_sigma1_list:
        nap = nap_sigma1[0]
        sigma1 = nap_sigma1[1]
        f_name = result_dir + '\\' + 'nap=' + str(nap) + '#sigma1=' + str(sigma1) + '#.txt'
        f_w = open(f_name, 'w')
        stop_number_list = []  # 用于记录每个nap对应的的停留点的个数
        for velocity_sigma in velocity_sigma_list:
            for parent, dirName, file_names in os.walk(data_dir):
                stops_count = 0
                for file_name in file_names:
                    data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
                    clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
                    # merged_clusters1 = Utility.merge_clusters(clusters)
                    # count = len(merged_clusters1)
                    count = len(clusters)
                    stops_count += count
                content = str(velocity_sigma) + ',' + str(stops_count) + ',\n'  # 每行记录表示格式为<nap,stops_count,>
                f_w.write(content)
                stop_number_list.append(stops_count)
        pl.plot(velocity_sigma_list, stop_number_list, 'vg-')
        pl.savefig(result_dir + '\\nap=' + str(nap) + ' sigma1=' + str(sigma1) + '.png', dpi=400)
        pl.close()
        f_w.close()


#parameter sigma_ma参数选择的配套函数，根据得到的txt结果绘制图片
def parameter_sigma_ma_plot(result_dir):
    plot_arr = []
    plot_model = ['^b--', '.r:', '*k--', 'vb:', '*k:']
    plot_count = 0
    max_sigma_ma = 0
    min_sigma_ma = 1000
    max_stops_num = 0
    min_stops_num = 1000
    legend_note_list = []
    for parent, dirName, file_names in os.walk(result_dir):
        for file_name in file_names:
            nap = file_name.split('#')[0].split('=')[1]
            sigma1 = file_name.split('#')[1].split('=')[1]
            note = r'nap=' + str(nap) + r'$,\sigma_1=$' + str(sigma1)
            legend_note_list.append(note)
            f_r = open(result_dir + '\\' + file_name, 'r')
            sigma_ma_list = []
            stop_number_list = []
            for line in f_r.xreadlines():
                sigma_ma = float(line.split(',')[0])
                num = float(line.split(',')[1])
                sigma_ma_list.append(sigma_ma)
                stop_number_list.append(num)
                if sigma_ma > max_sigma_ma:
                    max_sigma_ma = sigma_ma
                if sigma_ma < min_sigma_ma:
                    min_sigma_ma = sigma_ma
                if num > max_stops_num:
                    max_stops_num = num
                if num < min_stops_num:
                    min_stops_num = num
            p, = pl.plot(sigma_ma_list, stop_number_list, plot_model[plot_count])
            plot_count += 1
            plot_arr.append(p)
    pl.xlabel(r'$\sigma_{MA}$')
    pl.ylabel('number of stops')
    sigma_ma_interval = (max_sigma_ma - min_sigma_ma) / 10
    num_interval = (max_stops_num - min_stops_num) / 10
    pl.xlim(min_sigma_ma - sigma_ma_interval, max_sigma_ma + sigma_ma_interval)
    pl.ylim(min_stops_num - num_interval, max_stops_num + num_interval)
    pl.legend((plot_arr[0], plot_arr[1], plot_arr[2], plot_arr[3]),
              (legend_note_list[0], legend_note_list[1], legend_note_list[2], legend_note_list[3]),
              loc='best', numpoints=1, ncol=1)
    pl.savefig(result_dir + '\\result.png', dpi=600)
    pl.savefig(u"C:\\Users\\Administrator\\Desktop\\Figure 1\\sigma_ma_choose.png", dpi=200)
    pl.show()


# 选用不同轨迹点个数的轨迹，测试算法的运行时间，以此来对比不同算法的运算效率
def compare_efficiency(data_dir, txt_save_file, png_save_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    f_w = open(txt_save_file, 'w')
    velocity_sigma = 0.5
    num_limit = 71
    optimal_sigma = 0.5
    for parent, dirName, file_names in os.walk(data_dir):
        for file_name in file_names:
            print file_name + '\n'
            begin_time = time.time()   #记录开始时间
            content = file_name + ','
            data = Utility.read_geolife_data_file(data_dir + '\\' + file_name)
            compressed_data = Utility.data_preprocessing(data)
            partial_dist_list = Utility.caculate_adjacent_dist_list(compressed_data, num_limit)
            Utility.normalize_adjacent_dist_list(partial_dist_list)
            stability_list = Utility.calculate_stability(compressed_data, num_limit)
            potential_list = Utility.calculate_density_value(partial_dist_list, stability_list, optimal_sigma,
                                                             velocity_sigma)
            # 备份
            temp_potential_list = []
            for i in range(len(potential_list)):
                temp_potential_list.append(potential_list[i])
            # 画出potential的统计图  采用备份数据，函数里面需要排序
            potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)
            density_dist_list = range(len(compressed_data))
            dfc.add_attributes(compressed_data, potential_list,
                               density_dist_list)  # 完善了数据的potential和dist 属性
            clusters_dic = dbscan_with_datafield(compressed_data, potential_threshold,
                                                 num_limit)  # 返回的clusters 是字典型的
            clusters = []
            for id, points in clusters_dic.iteritems():
                if len(points) > 0:
                    clusters.append(points)
            clusters.sort(key=lambda pp: pp[0].time)
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
            png_file = png_save_dir + '\\' + file_name + '_' + 'nap_' + \
                       str(num_limit) + 'distSigma_' + str(optimal_sigma) + '.png'
            pl.savefig(png_file)
            pl.close()
            end_time = time.time()
            time_gap = end_time - begin_time
            point_number = len(compressed_data)
            content += str(point_number) + ','
            content += str(time_gap) + ',\n'
            f_w.write(content)
    f_w.close()


# observe the stop positions of a given user, input the user_id, plot the stop positions
def user_stops_observation(user_id):
    nap = 31
    sigma1 = 0.3
    velocity_sigma = 0.5
    begin_time = time.time()
    conn = MySQLdb.connect(host='localhost', user='root', passwd='root', db='stdatamining', charset='utf8')
    f_w = open("E:\\Science\\" + str(user_id) + "_stop_positions.txt", 'w')
    traid_select_sql = "SELECT traid FROM owndata_trajectory WHERE objid=" + str(user_id) + ';'
    cur = conn.cursor()
    cur.execute(traid_select_sql)
    results = cur.fetchall()
    traid_list = []
    for record in results:
        traid_list.append(record[0])
    start_points = []
    end_points = []
    stop_positions = []
    lon_low_limit = 360
    lon_high_limit = 0
    lat_low_limit = 360
    lat_high_limit = 0
    for traid in traid_list:
        print traid
        data = []
        poi_curve_lon = []  # 用于将所有的POI用虚线连接起来，包括起始点，路途中间的停留点
        poi_curve_lat = []
        points_select_sql = "SELECT longitude,latitude,time_date FROM owndata_point WHERE traid=" \
                            + str(traid) + "  ORDER BY time_date;"
        cur.execute(points_select_sql)
        point_records = cur.fetchall()
        for record in point_records:
            point = TrackPoint(float(record[0]), float(record[1]), long(Utility.convert_to_milsecond(str(record[2]))))
            data.append(point)
        if len(data) < 500:
            continue
        f_w.write(str(traid) + '\n')
        f_w.write('start position:' + str(data[0].lon) + ',' + str(data[0].lat) + ',\n')
        f_w.write('end position:' + str(data[len(data)-1].lon) + ',' + str(data[len(data)-1].lat) + ',\n')
        start_points.append(data[0])
        end_points.append(data[len(data)-1])
        poi_curve_lon.append(data[0].lon)   #保存第一个起始点的经纬度
        poi_curve_lat.append(data[0].lat)
        clusters = dbscan_process(data, velocity_sigma, nap, sigma1)  # 用我们的DBSCAN算法进行处理,得到停留点
        clusters = Utility.merge_clusters(clusters)
        clusters = Utility.clusters_refinement(clusters)
        f_w.write(str(len(clusters)) + '\n')
        for cluster in clusters:
            point_len = len(cluster)
            lon_sum = 0
            lat_sum = 0
            for point in cluster:
                lon_sum += point.lon
                lat_sum += point.lat
            lon_avg = lon_sum / point_len
            lat_avg = lat_sum / point_len
            stop_positions.append(TrackPoint(lon_avg, lat_avg, 0))
            poi_curve_lon.append(lon_avg)
            poi_curve_lat.append(lat_avg)
            f_w.write(str(lon_avg) + ',' + str(lat_avg) + ',\n')
        poi_curve_lon.append(data[len(data) - 1].lon)
        poi_curve_lat.append(data[len(data) - 1].lat)
        if lon_low_limit > min(poi_curve_lon):
            lon_low_limit = min(poi_curve_lon)
        if lon_high_limit < max(poi_curve_lon):
            lon_high_limit = max(poi_curve_lon)
        if lat_low_limit > min(poi_curve_lat):
            lat_low_limit = min(poi_curve_lat)
        if lat_high_limit < max(poi_curve_lat):
            lat_high_limit = max(poi_curve_lat)
        pl.plot(poi_curve_lon, poi_curve_lat, '.k--', markersize=4)
    stop_lon = [ele.lon for ele in stop_positions]
    stop_lat = [ele.lat for ele in stop_positions]
    l1, = pl.plot(stop_lon, stop_lat, 'xb', markersize=12)
    start_lon = [ele.lon for ele in start_points]
    start_lat = [ele.lat for ele in start_points]
    l2, = pl.plot(start_lon, start_lat, 'dg')
    end_lon = [ele.lon for ele in end_points]
    end_lat = [ele.lat for ele in end_points]
    l3, = pl.plot(end_lon, end_lat, '>r')
    plot_arr = [l1, l2, l3]
    lon_interval = (lon_high_limit - lon_low_limit) / 15
    lat_interval = (lat_high_limit - lat_low_limit) / 15
    pl.xlim(lon_low_limit-lon_interval, lon_high_limit+lon_interval)
    pl.ylim(lat_low_limit-lat_interval, lat_high_limit+lat_interval)
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.legend((plot_arr[1], plot_arr[0], plot_arr[2]), ('start position', 'stop position', 'end position'),
              loc='best', numpoints=1, ncol=1, frameon=False)
    pl.savefig(u"C:\\Users\Administrator\Desktop\\" + str(user_id) + "_positions.png", dpi=1000)
    f_w.close()
    cur.close()
    conn.close()
    end_time = time.time()
    print 'consume time: =' + str((end_time-begin_time)/60) + 'minutes'
    pl.show()
    pl.close()


# the main function
def main(file_name):
    nap = 51
    sigma1 = 0.3
    velocity_sigma = 0.5
    start_time = time.time()
    # show_track.show_track(file_name)
    # data = Utility.read_data_file(file_name)  # 注意修改数据文件时要修改读文件程序
    data = Utility.read_geolife_data_file(file_name)
    clusters = dbscan_process(data, velocity_sigma, nap, sigma1)
    # merged_clusters1 = Utility.merge_clusters(clusters)
    merged_clusters1 = clusters  #不采用合并
    merged_clusters = Utility.clusters_refinement(merged_clusters1)
    end_time = time.time()
    time_gap = end_time - start_time
    print 'consumed time is ' + str(time_gap) + ' seconds'
    Utility.print_clusters_info(merged_clusters)
    Utility.show_clusters(merged_clusters)
    Utility.show_clusters_trajectory(merged_clusters, data)


# file_name = u"D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_022\geolife_022_2009-07-09.txt"
# file_name = 'D:\\Python\\PaperPy\\data_filed_cluster\\data\\IMIS_3_DAY_129.txt'
# file_name = "D:\\Python\\PaperPy\\DataOperation\\data\\geolife_data\\observe\\342.txt"
# file_name = "D:\\Python\\PaperPy\\DataOperation\\data\\own_data\\11_26_am_12_50.txt"
# file_name = u"D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN" \
#             u"\compare_experiment\\new_DBSCAN\geolife_experiment\geolife_subtrajectory2\\90.txt"
# Utility.distance_rate(file_name)

# main(file_name)
# data_dir = u"D:\\Python\PaperPy\\data_filed_cluster\\compare_experiment\\new_DBSCAN\\own_data_experiment\\own_data"
# txt_save_dir = u"D:\Python\PaperPy\data_filed_cluster\compare_experiment\new_DBSCAN\own_data_experiment\\own_data_txt_result"
# png_save_dir = u"D:\Python\PaperPy\data_filed_cluster\compare_experiment\new_DBSCAN\own_data_experiment\\own_data_png_result"
# parameter_group_experiment(data_dir, txt_save_dir, png_save_dir)

# data_dir1 = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
#            u'\compare_experiment\\new_DBSCAN\parameter_compare_data'
# # dist_sigmas1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,1.4, 1.5]
# dist_sigmas1 = [0.1, 0.3, 0.5, 0.7, 1.1]
# parameter_choose(data_dir1, dist_sigmas1)
#
# data_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
#            u'\compare_experiment\\new_DBSCAN\efficient_compare\data'
# txt_save_file = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
#                 u'\compare_experiment\\new_DBSCAN\efficient_compare\\result.txt'
# png_save_dir = u'D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN' \
#                u'\compare_experiment\\new_DBSCAN\efficient_compare\png_result'

 # compare_efficiency(data_dir, txt_save_file, png_save_dir)

# traid_list = [400008, 400009, 400007]
# for traid in traid_list:
#     user_stops_observation(traid)

# user_stops_observation(500001)


# data_dir = u"D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\new_DBSCAN\\parameter_choose\\data_5"
# result_dir = u"D:\\Python\\PaperPy\\data_filed_cluster\\compare_experiment\\new_DBSCAN\\parameter_choose\\nap_sse\\result"
# parameter_nap_choose(data_dir, result_dir)
# parameter_nap_plot(result_dir)

# parameter_sigma1_choose(data_dir, result_dir)
# parameter_sigma_plot(result_dir)

# parameter_sigma_ma_choose(data_dir, result_dir)
# parameter_sigma_ma_plot(result_dir)

# sigma_sse_plot(result_dir)
# nap_sse_plot(result_dir)




