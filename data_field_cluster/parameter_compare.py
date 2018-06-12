# scoding:utf-8
import pylab as pl
import data_field_cluster as dfc
import numpy as np
import show_track
import Utility


# def compare_num_limit(min_num, max_num, dist_2_array, velocity_list, optimal_sigma, velocity_sigma):
def compare_num_limit(min_num, max_num, dist_2_array, velocity_list, optimal_sigma, velocity_sigma, compressed_data):
    colors = ['r', 'b', 'g', 'y', 'k']
    colors_len = len(colors)
    num_interval = (max_num - min_num)/(colors_len-1)
    first_num = min_num
    color_index = 0
    plot_arr = []
    num_arr = []
    # pl.figure(figsize=(8, 6), dpi=200)
    # pl.subplot(1, 1, 1)
    while first_num <= max_num:
        num_arr.append(first_num)
        dist_2_array_copy = []
        for arr in dist_2_array:  # 备份起来，最后的优化处理会用到
            arr_list = []
            for ele in arr:
                arr_list.append(ele)
            dist_2_array_copy.append(arr_list)
        # 在进行密度计算时，距离矩阵也需要先归一化
        partial_dist_list = Utility.calculate_partial_dist(dist_2_array, first_num)
        Utility.normalize_dist_array(dist_2_array_copy, partial_dist_list)
        stability_list = Utility.calculate_stability(compressed_data, first_num)  # 采用距离方差的方法
        potential_list = Utility.calculate_potential_value(dist_2_array_copy, stability_list,
                                                           optimal_sigma, velocity_sigma, first_num)
        potential_list.sort(reverse=True)
        sequence_xlable = []
        max_potential = max(potential_list)
        total_len = len(potential_list)

        for i in range(total_len):
            potential_list[i] = float(potential_list[i] / max_potential)
            sequence_xlable.append(float(i*1.0/total_len))
        l, = pl.plot(sequence_xlable, potential_list, colors[color_index % colors_len])
        plot_arr.append(l)
        pl.setp(l, markersize=3)
        pl.xlabel('normalized sequence')
        pl.ylabel('normalized density')
        first_num += num_interval
        color_index += 1
    pl.legend((plot_arr[0], plot_arr[1], plot_arr[2], plot_arr[3], plot_arr[4]),
              (r'Nap=' + str(num_arr[0]), r'Nap=' + str(num_arr[1]),
               r'Nap=' + str(num_arr[2]), r'Nap=' + str(num_arr[3]),
               r'Nap=' + str(num_arr[4])
               ), loc='best', numpoints=1)
    # pl.show()
    pl.savefig(u"C:\\Users\Administrator\Desktop\\1.png", dpi=1000)


def compare_distance_sigma(min_sigma, max_sigma, dist_2_array, velocity_list, velocity_sigma, num_limit):
    colors = ['r', 'b', 'g', 'y', 'k', 'c']
    colors_len = len(colors)
    sigma_interval = (max_sigma - min_sigma) / (len(colors)-1)
    first_sigma = min_sigma
    color_index = 0
    plot_arr = []
    sigma_arr = []
    while first_sigma <= max_sigma:
        dist_2_array_copy = []
        for arr in dist_2_array:  # 备份起来，最后的优化处理会用到
            arr_list = []
            for ele in arr:
                arr_list.append(ele)
            dist_2_array_copy.append(arr_list)
        # 在进行密度计算时，距离矩阵也需要先归一化
        partial_dist_list = Utility.calculate_partial_dist(dist_2_array, num_limit)
        Utility.normalize_dist_array(dist_2_array_copy, partial_dist_list)
        potential_list = Utility.calculate_potential_value(dist_2_array_copy, velocity_list,
                                                           first_sigma, velocity_sigma, num_limit)
        potential_list.sort(reverse=True)
        sequence_xlable = []
        max_potential = max(potential_list)
        total_len = len(potential_list)
        for i in range(total_len):
            potential_list[i] = float(potential_list[i] / max_potential)
            sequence_xlable.append(float(i * 1.0 / total_len))
        l, = pl.plot(sequence_xlable, potential_list, colors[color_index % colors_len])
        plot_arr.append(l)
        sigma_arr.append(first_sigma)
        pl.setp(l, markersize=3)
        first_sigma += sigma_interval
        color_index += 1
    pl.legend((plot_arr[0], plot_arr[1], plot_arr[2], plot_arr[3], plot_arr[4], plot_arr[5]),
              (r'$\sigma_1=$' + str(sigma_arr[0]), r'$\sigma_1=$' + str(sigma_arr[1]),
               r'$\sigma_1=$' + str(sigma_arr[2]), r'$\sigma_1=$' + str(sigma_arr[3]),
               r'$\sigma_1=$' + str(sigma_arr[4]), r'$\sigma_1=$' + str(sigma_arr[5])
               ), loc='best', numpoints=1)
    pl.xlabel('normalized sequence')
    pl.ylabel('normalized density')
    # pl.title('the density influence of sigma')
    # pl.show()
    pl.savefig(u"C:\\Users\Administrator\Desktop\\2.png", dpi=1000)


def test_num_limit(file_name, min_num, max_num, optimal_sigma, velocity_sigma):
    show_track.show_track(file_name)
    # data = Utility.read_data_file(file_name)                 # 注意修改数据文件时要修改读文件程序
    data = Utility.read_geolife_data_file(file_name)
    compressed_data = Utility.data_preprocessing(data)
    dist_2_array = Utility.calculate_dist_2_array(compressed_data)
    dfc.normalize_point_velocity(compressed_data)            # 速度归一化
    velocity_list = [point.velocity for point in compressed_data]
    compare_num_limit(min_num, max_num, dist_2_array, velocity_list, optimal_sigma, velocity_sigma, compressed_data)


def test_distance_sigma(file_name, min_sigma, max_sigma, velocity_sigma, num_limit):
    show_track.show_track(file_name)
    # data = Utility.read_data_file(file_name)                 # 注意修改数据文件时要修改读文件程序
    data = Utility.read_geolife_data_file(file_name)
    compressed_data = Utility.data_preprocessing(data)
    dist_2_array = Utility.calculate_dist_2_array(compressed_data)
    dfc.normalize_point_velocity(compressed_data)  # 速度归一化
    # velocity_list = [point.velocity for point in compressed_data]
    # compare_distance_sigma(min_sigma, max_sigma, dist_2_array, velocity_list, velocity_sigma, num_limit)
    stability_list = Utility.calculate_stability(compressed_data, num_limit)  #采用距离方差的方法
    compare_distance_sigma(min_sigma, max_sigma, dist_2_array, stability_list, velocity_sigma, num_limit)

# f_name = u"D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_022\geolife_022_2009-07-16.txt"
f_name = "D:\Python\PaperPy\data_filed_cluster\experience_result\\new experiment\\NewDBSCAN\compare_experiment" \
         "\CB-SMotT\geolife_experiment\geolife_subtrajectory2\\90.txt"
# f_name = 'data\IMIS_3_DAY_152.txt'
# vel_sigma = 0.5
# adj_num = 51

test_num_limit(f_name, 20, 100, 0.3, 0.5)
# test_distance_sigma(f_name, 0.1, 0.8, vel_sigma, adj_num)

