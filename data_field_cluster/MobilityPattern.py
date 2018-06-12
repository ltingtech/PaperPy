# _*_coding:utf-8_*_
import Utility
import data_field_cluster as dfc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
from TrackPoint import TrackPoint
import re
import math


# 返回分段的索引值
def get_track_segment(track_data):
    result = []
    if len(track_data) > 0:
        i = 1
        result.append(0)
        while i < len(track_data):
            if track_data[i].time-track_data[i-1].time > 3600*1000:
                result.append(i)
            i += 1
        result.append(len(track_data)-1)
    return result


def calculate_day_poi(file_name):
    data = Utility.read_geolife_data_file(file_name)
    segment_index = get_track_segment(data)
    # segment_index = [0, len(data)-1]
    result = []
    if len(segment_index) >= 2:
        i = 1
        while i < len(segment_index):
            segment_track = data[segment_index[i-1]:segment_index[i]]
            track_poi = get_poi(segment_track)
            result += track_poi
            i += 1
    return result


def get_poi(data):
    result = []
    if len(data) > 500:
        compressed_data = Utility.data_preprocessing(data)
        num_limit = 51
        dist_2_array = Utility.calculate_dist_2_array(compressed_data)
        dist_2_array_copy = []
        for arr in dist_2_array:  # 备份起来，最后的优化处理会用到
            arr_list = []
            for ele in arr:
                arr_list.append(ele)
            dist_2_array_copy.append(arr_list)
        partial_dist_list = Utility.calculate_partial_dist(dist_2_array, num_limit)
        Utility.normalize_dist_array(dist_2_array, partial_dist_list)
        optimal_sigma = 0.3
        dfc.normalize_point_velocity(data)
        velocity_list = [point.velocity for point in compressed_data]
        potential_list = Utility.calculate_potential_value(dist_2_array, velocity_list, optimal_sigma, num_limit)
        temp_potential_list = []
        for i in range(len(potential_list)):
            temp_potential_list.append(potential_list[i])
        potential_threshold = dfc.calculate_potential_threshold(temp_potential_list)
        dfc.refresh_dist_array(compressed_data, dist_2_array)
        density_dist_list = dfc.calculate_density_distance(dist_2_array, potential_list)
        dfc.add_attributes(compressed_data, potential_list, density_dist_list)
        temp_distance = []
        for i in range(len(density_dist_list)):
            temp_distance.append(density_dist_list[i])
        dist_threshold = dfc.calculate_dist_threshold(temp_distance)
        centre_potential = []
        centre_dist = []
        centre_index_list = []
        if potential_threshold > 0 and dist_threshold > 0:
            for i in range(len(density_dist_list)):
                if potential_list[i] > potential_threshold and density_dist_list[i] > dist_threshold:
                    centre_potential.append(potential_list[i])
                    centre_dist.append(density_dist_list[i])
                    centre_index_list.append(i)
        else:
            print 'there are something wrong with the threshold'

        # stop_point_list = dfc.get_stop_position(compressed_data, centre_index_list)
        centre_index_merge = dfc.merge_stop_position(dist_2_array_copy, centre_index_list, potential_list)
        new_centre_index_list = dfc.refine_stop_position(dist_2_array_copy, centre_index_merge, compressed_data)
        new_stop_point_list = dfc.get_stop_position(compressed_data, new_centre_index_list)
        result = new_stop_point_list
    return result


def calculate_all_poi(file_dir, save_file):
    result = []
    for parent, dirName, file_names in os.walk(file_dir):
        for file_name in file_names:
            print u'当前处理文件：%s' % file_name
            f = file_dir + '\\' + file_name
            day_poi = calculate_day_poi(f)
            result += day_poi
    f = open(save_file, 'w')
    for point in result:
        f.write(str(point.lon) + ',' + str(point.lat) + ',' +
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(point.time / float(1000))) + ',\n')
    print 'good!'


def normalize_list(data_list):
    max_data = max(data_list)
    min_data = min(data_list)
    gap = max_data-min_data
    for i, data in enumerate(data_list):
        data_list[i] = (data-min_data)/float(gap)


def show_pattern(poi_file):
    data = [[float(ele.split(',')[0]), float(ele.split(',')[1]), ele.split(',')[2]] for ele in open(poi_file, 'r')]
    if len(data) > 0:
        x_lon = []
        y_lat = []
        z_time = []
        for ele in data:
            x_lon.append(ele[0])
            y_lat.append(ele[1])
            z_time.append(int(time.strptime(ele[2], "%Y-%m-%d %H:%M:%S").tm_hour)/float(24))
            # hour = int(time.strptime(ele[2], "%Y-%m-%d %H:%M:%S").tm_hour)
            # if hour < 7:
            #     z_time.append(float(6)/24)
            # else:
            #     if 7 <= hour < 10:
            #         z_time.append(float(9) / 24)
            #     else:
            #         if 10 <= hour < 12:
            #             z_time.append(float(11)/24)
            #         else:
            #             if 12 <= hour <= 14:
            #                 z_time.append(float(13)/24)
            #             else:
            #                 if 16 <= hour <= 18:
            #                     z_time.append(float(17)/24)
            #                 else:
            #                     if 18 <= hour <= 20:
            #                         z_time.append(float(19)/24)
            #                     else:
            #                         z_time.append(float(22)/24)
        normalize_list(x_lon)
        normalize_list(y_lat)
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_lon, y_lat, z_time, c='g', marker='o')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('time')
        plt.show()
    else:
        print u'没有检测到热点区域'


# ***************************************
# ********以下实验为学位论文的内容*******
# ***************************************
# 从txt数据集中解析出所有的停留点、出发地、结束地
def extract_all_position(f_name):
    position_list = []
    f_r = open(f_name, 'r')
    line = f_r.readline()
    while line != '':
        traid = line.split('\n')[0]
        count = 0
        while count < 2:             #先读取两行，包括出发地和结束地
            line = f_r.readline()
            content = re.split(':|,', line)
            position = TrackPoint(float(content[1]), float(content[2]), 0)
            position_list.append(position)
            count += 1
        line = f_r.readline()
        stop_num = int(line.split('\n')[0])
        count = 0
        while count < stop_num:
            line = f_r.readline()
            content = re.split(',', line)
            position = TrackPoint(float(content[0]), float(content[1]), 0)
            position_list.append(position)
            count += 1
        line = f_r.readline()
    return position_list


# 从txt数据集中解析出所有的停留区域
def extract_stop_position(f_name):
    position_list = []
    f_r = open(f_name, 'r')
    line = f_r.readline()
    while line != '':
        traid = line.split('\n')[0]
        count = 0
        while count < 2:             #先读取两行，包括出发地和结束地
            line = f_r.readline()
            count += 1
        line = f_r.readline()
        stop_num = int(line.split('\n')[0])
        count = 0
        while count < stop_num:
            line = f_r.readline()
            content = re.split(',', line)
            position = TrackPoint(float(content[0]), float(content[1]), 0)
            position_list.append(position)
            count += 1
        line = f_r.readline()
    return position_list


# 计算距离矩阵
def calculate_dist_arr(data):
    dist_arr = []
    data_len = len(data)
    for i in range(data_len):
        dist_list = [0]
        for j in range(i+1, data_len):
            dist_list.append(Utility.distance_calculate(data[i], data[j]))
        dist_arr.append(dist_list)
    return dist_arr


# 对距离进行归一化
def normalize_dist_arr(dist_arr, high_limit):
    for i in range(len(dist_arr)):
        for j in range(len(dist_arr[i])):
            if dist_arr[i][j] >= high_limit:
                dist_arr[i][j] = 1
            else:
                dist_arr[i][j] /= high_limit


# 计算点的属性，包括potential属性和distance属性
def calculate_attributes(data, dist_arr, dist_thrsd):
    data_len = len(data)
    max_potential = 0
    normalize_dist_arr(dist_arr, dist_thrsd)
    # 完善potential属性
    for idx in range(data_len):
        potential = 0
        for i in range(idx):
            potential += math.exp((-1) * math.pow(dist_arr[i][idx-i] / 0.3, 2))
            # if dist_arr[i][idx - i] <= dist_thrsd:
            #     potential += 1
        for j in range(idx, data_len):
            potential += math.exp((-1) * math.pow(dist_arr[idx][j-idx] / 0.3, 2))
            # if dist_arr[idx][j - idx] <= dist_thrsd:
            #     potential += 1
        if potential > max_potential:
            max_potential = potential
        data[idx].potential = potential
    # 完善distance属性
    for idx in range(data_len):
        current_potential = data[idx].potential
        if current_potential == max_potential:
            data[idx].distance = 1
            continue
        min_distance = 100000000
        for i in range(idx):
            if current_potential < data[i].potential:
                if dist_arr[i][idx - i] < min_distance:
                    min_distance = dist_arr[i][idx - i]
        for j in range(idx+1, data_len):
            if current_potential < data[j].potential:
                if dist_arr[idx][j-idx] < min_distance:
                    min_distance = dist_arr[idx][j - idx]
        data[idx].distance = min_distance


# 提取热点区域点
def extract_hot_spot(data):
    dist_arr = calculate_dist_arr(data)
    dist_high_limit = 100
    calculate_attributes(data, dist_arr, dist_high_limit)
    dist_thrsd = 0.9
    potential_thrsd = 10
    hot_spot_list = []
    for point in data:
        if point.distance >= dist_thrsd and point.potential >= potential_thrsd:
            hot_spot_list.append(point)
    return hot_spot_list


# 在路网中画出热点区域
def plot_hot_spot(file_name):
    data = extract_stop_position(file_name)
    hot_spot_list = extract_hot_spot(data)
    start_positions = []
    end_positions = []
    f_r = open(file_name, 'r')
    line = f_r.readline()
    while line != '':
        route = []
        traid = line.split('\n')[0]
        # 先读取两行，包括出发地和结束地
        line = f_r.readline()
        content = re.split(':|,', line)
        s_position = TrackPoint(float(content[1]), float(content[2]), 0)
        start_positions.append(s_position)
        route.append(s_position)
        line = f_r.readline()
        content = re.split(':|,', line)
        e_position = TrackPoint(float(content[1]), float(content[2]), 0)
        end_positions.append(e_position)
        line = f_r.readline()
        stop_num = int(line.split('\n')[0])
        count = 0
        while count < stop_num:
            line = f_r.readline()
            content = re.split(',', line)
            position = TrackPoint(float(content[0]), float(content[1]), 0)
            route.append(position)
            count += 1
        route.append(e_position)
        route_lon = [p.lon for p in route]
        route_lat = [p.lat for p in route]
        plt.plot(route_lon, route_lat, '.k--', markersize=4)
        line = f_r.readline()
    f_r.close()
    # 画出所有的停留点
    stop_lon = [p.lon for p in data]
    stop_lat = [p.lat for p in data]
    l1, = plt.plot(stop_lon, stop_lat, 'xb', markersize=10)
    # 画出发点、结束点
    s_p_lon = [p.lon for p in start_positions]
    s_p_lat = [p.lat for p in start_positions]
    e_p_lon = [p.lon for p in end_positions]
    e_p_lat = [p.lat for p in end_positions]
    l2, = plt.plot(s_p_lon, s_p_lat, '<g')
    l3, = plt.plot(e_p_lon, e_p_lat, '>r')
    # 画出所有的热点区域所在地
    hot_spot_lon = [p.lon for p in hot_spot_list]
    hot_spot_lat = [p.lat for p in hot_spot_list]
    l4, = plt.plot(hot_spot_lon, hot_spot_lat, '*y', markersize=16)
    #设置横纵坐标范围
    lon_high_limit = max(max(stop_lon), max(s_p_lon), max(e_p_lon))
    lon_low_limit = min(min(stop_lon), min(s_p_lon), min(e_p_lon))
    lat_high_limit = max(max(stop_lat), max(s_p_lat), max(e_p_lat))
    lat_low_limit = min(min(stop_lat), min(s_p_lat), min(e_p_lat))
    lon_interval = (lon_high_limit - lon_low_limit) / 15
    lat_interval = (lat_high_limit - lat_low_limit) / 15
    plt.xlim(lon_low_limit - lon_interval, lon_high_limit + lon_interval)
    plt.ylim(lat_low_limit - lat_interval, lat_high_limit + lat_interval)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    # plot_arr = [l1, l2, l3]
    plot_arr = [l1, l2, l3, l4]
    plt.legend((plot_arr[1], plot_arr[0], plot_arr[2], plot_arr[3]),
               ('start position', 'stop position', 'end position', 'hot-spot position'),
              loc='best', numpoints=1, ncol=1, frameon=False)
    # plt.legend((plot_arr[1], plot_arr[0], plot_arr[2]),
    #            ('start position', 'stop position', 'end position'),
    #            loc='best', numpoints=1, ncol=1, frameon=False)
    plt.savefig(u"C:\\Users\Administrator\Desktop\\result.png", dpi=400)
    plt.show()


# 将停留区域编码,并将轨迹转化为编码序列
def encode_stop_positions(data_file, code_save_file):
    data = extract_stop_position(data_file)
    hot_spot_list = extract_hot_spot(data)
    id_begin = '1000'
    position_code_dic = {}
    for point in hot_spot_list:
        position_code_dic[id_begin] = (point.lon, point.lat)
        id_begin = str(int(id_begin) + 1)
    all_position = extract_all_position(data_file)
    for position in all_position:
        is_encoded = False
        for id_key in position_code_dic.keys():
            id_value = position_code_dic[id_key]
            if Utility.distance_calculate(position, TrackPoint(id_value[0], id_value[1], 0)) <= 200:
                is_encoded = True
                break
            else:
                continue
        if not is_encoded:
            position_code_dic[id_begin] = (position.lon, position.lat)
            id_begin = str(int(id_begin) + 1)
    f_w = open(code_save_file, 'w')
    dic_len = len(position_code_dic)
    f_w.write(str(dic_len) + '\n')
    for k in sorted(position_code_dic.keys()):
        p = position_code_dic[k]
        content = k + ':' + str(p[0]) + ',' + str(p[1]) + ',\n'
        f_w.write(content)
    # 开始按轨迹依次进行编码
    f_r = open(data_file, 'r')
    line = f_r.readline()
    while line != '':
        traj_code = []
        traid = line.split('\n')[0]
        # 先读取两行，包括出发地和结束地
        line = f_r.readline()
        content = re.split(':|,', line)
        s_position = TrackPoint(float(content[1]), float(content[2]), 0)
        code = get_code(position_code_dic, s_position)
        traj_code.append(code)
        line = f_r.readline()
        content = re.split(':|,', line)
        e_position = TrackPoint(float(content[1]), float(content[2]), 0)  #结束点先慢点处理
        line = f_r.readline()
        stop_num = int(line.split('\n')[0])
        count = 0
        while count < stop_num:
            line = f_r.readline()
            content = re.split(',', line)
            position = TrackPoint(float(content[0]), float(content[1]), 0)
            code = get_code(position_code_dic, position)
            traj_code.append(code)
            count += 1
        # 结束点最后处理
        code = get_code(position_code_dic, e_position)
        traj_code.append(code)
        code_len = len(traj_code)
        if code_len >= 2 and traj_code[code_len - 1] == traj_code[code_len - 2]:
            del traj_code[code_len - 1]
        if len(traj_code) >= 2 and traj_code[0] == traj_code[1]:
            del traj_code[0]
        content = ','.join(traj_code)
        f_w.write(content + '\n')
        # 循环读
        line = f_r.readline()
    f_r.close()
    f_w.close()



# 根据位置确定编号
def get_code(code_dic, point):
    code_dist = ('00', 2000)
    for k in code_dic.keys():
        value_p = code_dic[k]
        dist = Utility.distance_calculate(point, TrackPoint(value_p[0], value_p[1], 0))
        if dist <= code_dist[1]:
            code_dist = (k, dist)    # 返回距离最近的区域的编码
        else:
            continue
    return code_dist[0]


# 利用所有的点画出决策图
def plot_decision_graph(file_name):
    data = extract_stop_position(file_name)
    dist_arr = calculate_dist_arr(data)
    dist_thrsd = 100
    calculate_attributes(data, dist_arr, dist_thrsd)
    potential_list = [p.potential for p in data]
    dist_list = [p.distance for p in data]
    plt.plot(potential_list, dist_list, 'og')
    plt.xlabel('density')
    plt.ylabel('distance')
    plt.ylim(0, 1.05)
    hotspot_potential = []
    hotspot_dist = []
    for p in data:
        if p.potential > 10 and p.distance > 0.9:
            hotspot_potential.append(p.potential)
            hotspot_dist.append(p.distance)
    plt.plot(hotspot_potential, hotspot_dist, '*r', markersize=12)
    plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\png\decision graph.png', dpi=200)
    plt.show()
    potential_list.sort(reverse=True)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_potential_list = Utility.calculate_conv(potential_list, kernel)
    plt.plot(smoothed_potential_list, )
    plt.xlabel('sequence')
    plt.ylabel('potential')
    plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\png\potential.png', dpi=200)
    plt.show()
    dist_list.sort(reverse=True)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_distance_list = Utility.calculate_conv(dist_list, kernel)
    plt.plot(smoothed_distance_list)
    plt.xlabel('sequence')
    plt.ylabel('distance')
    plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\png\distance.png', dpi=200)
    plt.show()


# Apriori算法主体
def apriori_algorithm(code_f, min_support, result_save_file):
    f_r = open(code_f, 'r')
    line = f_r.readline()
    count = int(line.split('\n')[0])
    while count > 0:
        f_r.readline()
        count -= 1
    line = f_r.readline()
    traj_sequence = []
    while line != '':
        sequence = re.split(',|\n', line)
        del sequence[len(sequence)-1]
        traj_sequence.append(sequence)
        line = f_r.readline()
    f_r.close()
    f_w = open(result_save_file, 'w')
    iterms_list = gen_single_iterm(traj_sequence, min_support)
    iterms_list_len = len(iterms_list)
    while iterms_list_len > 0:
        f_w.write(str(iterms_list_len) + '\n')
        # 每次进行下一次循环前先把频繁项集保存起来
        for iterm in iterms_list:
            content = ','.join(iterm[0]) + ',' + str(iterm[1]) + ',\n'
            f_w.write(content)
        iterms_list = apriori_gen(iterms_list)
        for sequence in traj_sequence:
            for iterm in iterms_list:
                num = contained_num(sequence, iterm[0])
                iterm[1] += num
        idx = len(iterms_list)-1
        while idx >= 0:
            if not iterms_list[idx][1] >= min_support:
                del iterms_list[idx]
            idx -= 1
        iterms_list_len = len(iterms_list)
    f_w.close()


# 判断一个项集先一个路径序列的次数
def contained_num(sequence, iterm_key):
    result = 0
    if len(iterm_key) > len(sequence):
        result = 0
    else:
        iterm_key_len = len(iterm_key)
        sequence_len = len(sequence)
        for i in range(sequence_len-iterm_key_len + 1):
            is_contain = True
            for j in range(iterm_key_len):
                if not sequence[i+j] == iterm_key[j]:
                    is_contain = False
                    break
            if is_contain:
                result += 1
    return result


# 产生所用的频繁 1-项集
def gen_single_iterm(traj_sequence, min_support):
    result = []
    temp_dic = {}
    for sequence in traj_sequence:
        for ele in sequence:
            if not temp_dic.has_key(ele):
                temp_dic[ele] = 1
            else:
                temp_dic[ele] += 1
    for k in temp_dic.keys():
        if temp_dic[k] >= min_support:
            result.append([[k], temp_dic[k]])
    return result


# 候选项集的产生
def apriori_gen(iterm_list):
    result = []
    for iterm_i in iterm_list:
        for iterm_j in iterm_list:
            if is_combinable(iterm_i, iterm_j):  #如果可以合并
                new_iterm_key = []
                for ele in iterm_i[0]:
                    new_iterm_key.append(ele)
                iterm_key_len = len(iterm_j[0])
                new_iterm_key.append(iterm_j[0][iterm_key_len-1])
                result.append([new_iterm_key, 0])
    return result


# 检查两个项集是否满足合并的要求
def is_combinable(iterm_i, iterm_j):
    result = True
    iterm_len = len(iterm_i[0])
    if iterm_len == 1:
        if iterm_i[0][0] == iterm_j[0][0]:
            result = False
    else:
        iterm_i_key = iterm_i[0]
        iterm_j_key = iterm_j[0]
        for i in range(1, len(iterm_i_key)):
            if not iterm_i_key[i] == iterm_j_key[i-1]:
                result = False
                break
    return result


# 画出频繁轨迹路径2,将所有的热点都画出来
def plot_frequent_pattern_2(code_file, frequent_pattern_file):
    f_r = open(code_file, 'r')
    line = f_r.readline()
    count = int(line.split('\n')[0])
    position_code_dic = {}
    while count > 0:
        line = f_r.readline()
        line_ele = re.split(':|,', line)
        k = line_ele[0]
        v = [line_ele[1], line_ele[2]]
        position_code_dic[k] = v
        count -= 1
    f_r.close()
    hot_spot_lon = []
    hot_spot_lat = []
    for k in ['1000', '1001', '1002', '1003', '1004', '1005']:
        hot_spot_lon.append(position_code_dic[k][0])
        hot_spot_lat.append(position_code_dic[k][1])
    f_r = open(frequent_pattern_file, 'r')
    line = f_r.readline()
    # fig = plt.figure(figsize=(10, 1))
    # fig_mode = [133, 132, 131]
    count = int(line.split('\n')[0])
    while count > 0:
        line = f_r.readline()
        positions_code_list = line.split(',')
        positions = []
        for i in range(len(positions_code_list) - 2):
            positions.append(position_code_dic[positions_code_list[i]])
        lon = [ele[0] for ele in positions]
        lat = [ele[1] for ele in positions]
        # ax = fig.add_subplot(fig_mode[count-1], autoscale_on=False)
        # plt.subplot(fig_mode[count - 1])
        # ax.plot(lon, lat, '.-.k')
        # ax.plot(hot_spot_lon, hot_spot_lat, 'r*', markersize=8)
        plt.plot(lon, lat, '.-.k')
        # plt.xscale('log', basex=2)
        l2 = plt.plot(hot_spot_lon, hot_spot_lat, 'r*', markersize=8)
        # plt.legend(l2, 'hot-spot position', loc='best', numpoints=1, ncol=1, frameon=False)
        plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\frequent_' + str(9-count) + '.png', dpi=200)
        count -= 1
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.close()
    f_r.close()

# 画出频繁轨迹路径，只画出经过的那几个热点区域
def plot_frequent_pattern(code_file, frequent_pattern_file):
    f_r = open(code_file, 'r')
    line = f_r.readline()
    count = int(line.split('\n')[0])
    position_code_dic = {}
    while count > 0:
        line = f_r.readline()
        line_ele = re.split(':|,', line)
        k = line_ele[0]
        v = [line_ele[1], line_ele[2]]
        position_code_dic[k] = v
        count -= 1
    f_r.close()
    f_r = open(frequent_pattern_file, 'r')
    line = f_r.readline()
    # fig = plt.figure()
    fig_mode = [223, 222, 221]
    count = int(line.split('\n')[0])
    while count > 0:
        line = f_r.readline()
        positions_code_list = line.split(',')
        positions = []
        for i in range(len(positions_code_list) - 2):
            positions.append(position_code_dic[positions_code_list[i]])
        lon = [ele[0] for ele in positions]
        lat = [ele[1] for ele in positions]
        # ax = fig.add_subplot(fig_mode[count-1], autoscale_on=False)
        # plt.subplot(fig_mode[count - 1])
        # ax.plot(lon, lat, '--kx')
        plt.plot(lon, lat, '--kx', markersize=12)
        # plt.xscale('log', basex=2)
        plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\frequent_' + str(7 - count) + '.png',
                    dpi=200)
        count -= 1
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.close()
    f_r.close()



    # plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\frequent pattern.png', dpi=600)
    # plt.show()


# 画出目标的路径网络
def plot_path_net(code_file):
    f_r = open(code_file, 'r')
    line = f_r.readline()
    count = int(line.split('\n')[0])
    position_code_dic = {}
    while count > 0:
        line = f_r.readline()
        line_ele = re.split(':|,', line)
        k = line_ele[0]
        v = [line_ele[1], line_ele[2]]
        position_code_dic[k] = v
        count -= 1
    line = f_r.readline()
    while line != '':
        k_list = re.split(',|\n', line)
        del k_list[len(k_list) - 1]
        lon_list = [position_code_dic[p][0] for p in k_list]
        lat_list = [position_code_dic[p][1] for p in k_list]
        plt.plot(lon_list, lat_list, '-k', linewidth=1)
        line = f_r.readline()
    f_r.close()
    region_lon = [position_code_dic[k][0] for k in position_code_dic.keys()]
    region_lat = [position_code_dic[k][1] for k in position_code_dic.keys()]
    l2 = plt.plot(region_lon, region_lat, '*b', markersize=10)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    # plt.legend(l2, 'stop postion', loc='lower left', numpoints=1, frameon=False)
    plt.savefig(u'D:\Python\PaperPy\data_filed_cluster\EduPaper\png\path_net.png', dpi=200)
    plt.show()
    plt.close()



# ***************************************


def main():
    data_dir = 'D:\Python\PaperPy\DataOperation\data\geolife_data\geolife_153_2'
    save_file = 'D:\Python\PaperPy\DataOperation\data\geolife_data\poi_153.txt'
    calculate_all_poi(data_dir, save_file)
    show_pattern(save_file)


# main()

file_name = u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\500001_stop_positions_31.txt'
code_file = u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\code.txt'
frequent_pattern_file = u'D:\Python\PaperPy\data_filed_cluster\EduPaper\data\\frequent_pattern.txt'
# extract_all_position(file_name)
# plot_decision_graph(file_name)
# plot_hot_spot(file_name)
# encode_stop_positions(file_name, code_file)
apriori_algorithm(code_file, 10, frequent_pattern_file)

# plot_frequent_pattern_2(code_file, frequent_pattern_file)

# plot_path_net(code_file)

