# _*_coding:utf-8_*_

import math
import pylab as pl
import matplotlib.pyplot as plt
from TrackPoint import TrackPoint
import numpy
import time
import datetime
import show_track

# 全局变量
marker_size = 9
marker_face_color = 'blue'
figure_size = (8, 6)
line_width = 2


def distance_calculate(point1, point2):
    lon1, lat1, lon2, lat2 = map(convert_to_radius, [point1.lon, point1.lat, point2.lon, point2.lat])  # 迭代函数
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371                 # 地球平均半径，单位为公里
    return round(c * r * 10000/10.0, 4)   # 单位为米


def convert_to_radius(value):
    return (math.pi/180)*value




def read_data_file(file_name):
    try:
        point_list = [TrackPoint(float(ele.split(',')[3]), float(ele.split(',')[4]), long(ele.split(',')[2]))
                      for ele in open(file_name, 'r')]
        return point_list
    except Exception as e:
        print e
        
# 为了方便geolife数据


def convert_to_milsecond(date_string):
    t = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    d = t.timetuple()
    ss = str(int(time.mktime(d)*1000))  # 毫秒
    return ss


# 读取geolife数据
def read_geolife_data_file(file_name):
    try:
        point_list = [TrackPoint(float(ele.split(',')[0]), float(ele.split(',')[1]),
                                 long(convert_to_milsecond(ele.split(',')[2]))) for ele in open(file_name, 'r')]
        return point_list
    except Exception as e:
        print e


# 读取手机搜集的数据
def read_own_data_file(file_name):
    point_list = []
    try:
        f = open(file_name, 'r')
        for line in f:
            eleArr = line.split(',')
            lon = eleArr[1]  # 注意经纬度不能读反了
            lat = eleArr[0]
            t = convert_to_milsecond(eleArr[2].split('.')[0].replace('T', ' '))
            p = TrackPoint(float(lon), float(lat), long(t))
            p.time_str = eleArr[2]
            point_list.append(p)
    except Exception as e:
        print e
    return point_list


def show_cluster(cluster_list):
    colors = ['or', 'ob', 'og', 'oy', 'ok']
    length = len(colors)
    count = 0
    for cluster in cluster_list:
        lon_list = [point.lon for point in cluster]
        lat_list = [point.lat for point in cluster]
        color = colors[count%length]
        pl.plot(lon_list, lat_list, color)
        count += 1
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.show()


# 计算距离进行插值时考虑的距离间隔
def mean_dist_interval(point_list):
    i = 0
    interval_list = []
    while i < len(point_list)-1:
        interval_list.append(distance_calculate(point_list[i], point_list[i+1]))
        i += 1
        interval_list.sort(reverse=True)
    kernel = generate_gaus_kernel(4)
    smoothed_interval_list = calculate_conv(interval_list, kernel)
    curvature_index = calculate_curvature3(smoothed_interval_list, False)
    if curvature_index > 0:
        threshold = interval_list[curvature_index]
        if threshold <= 0:
            return -1
        dist_sum = 0
        count = 0
        for dist in interval_list:
            if dist < threshold:
                dist_sum += dist
                count += 1
        mean_dist = dist_sum/float(count)
        return mean_dist*2
        # return threshold
    else:
        return -1


def data_preprocessing_no_compress(point_list):
    new_point_list = []
    for i in range(len(point_list)-1):
        if point_list[i].time == point_list[i + 1].time:
            continue
        dist = distance_calculate(point_list[i], point_list[i + 1])
        if dist / ((point_list[i + 1].time - point_list[i].time) / float(1000)) > 30:  # 速度上限取30
            continue
        new_point_list.append(point_list[i])
    return new_point_list

    '''

    mean_dist = mean_dist_interval(point_list)
    if mean_dist <= 0:
        return point_list
    else:
        for i in range(len(point_list) - 1):
            if point_list[i].time == point_list[i + 1].time:
                continue
            dist = distance_calculate(point_list[i], point_list[i + 1])

            if dist >= mean_dist:
                if dist / ((point_list[i + 1].time - point_list[i].time) / float(1000)) > 30:  # 速度上限取30
                    continue

                else:
                    num1 = 2
                    num2 = int(dist / mean_dist)
                    num = max(num1, num2)
                    new_point_list.append(point_list[i])
                    lon = point_list[i].lon
                    lat = point_list[i].lat
                    time1 = point_list[i].time
                    lon_step = (point_list[i + 1].lon - point_list[i].lon) / num
                    lat_step = (point_list[i + 1].lat - point_list[i].lat) / num
                    time_step = (point_list[i + 1].time - point_list[i].time) / num
                    for k in range(1, num):
                        p = TrackPoint(lon + k * lon_step, lat + k * lat_step, time1 + k * time_step)
                        p.is_interpolated = True
                        new_point_list.append(p)


            else:
                new_point_list.append(point_list[i])

        return new_point_list
        '''


def data_preprocessing(point_list):  # 数据的预处理，包括压缩和插值
        new_point_list = data_preprocessing_no_compress(point_list)
        # new_point_list_len = len(new_point_list)
        # print u'补全后的点数量： %d' % new_point_list_len
        # if new_point_list_len < 2000:
        #     return new_point_list
        # sample_interval = int(math.log(new_point_list_len, math.e))  # 新的采样率
        # print u'采样率'
        # print sample_interval
        # if sample_interval < 1:
        #     sample_interval = 1
        # result = []
        # ii = 0
        # while ii < new_point_list_len:
        #     result.append(new_point_list[ii])
        #     ii += sample_interval
        #
        # print u'原始数据点个数'
        # print len(point_list)
        # print u'插值后数据点个数'
        # print len(new_point_list)
        #
        # print u'最终采样后的点个数为： %d' % len(result)
        #
        # return result
        return new_point_list



# 计算距离矩阵和轨迹点的速度
def calculate_dist_2_array(point_list):
    result = []         # 返回结果是一个二维数组
    list_len = len(point_list)
    count = 0
    for i, point in enumerate(point_list):
        count += 1
        dist_ele = [0]   # 每个点都与后面的点计算距离
        for j in range(i+1, list_len):
            if point_list[j].time >= point_list[i].time:
                dist = distance_calculate(point_list[i], point_list[j])
                if j == i+1 and point_list[j].time > point_list[i].time:
                    point_list[i].velocity = dist/float((point_list[j].time-point_list[i].time)/float(1000))  # 速度：米/秒
                if point_list[j].time == point_list[i].time:
                    point_list[i].velocity = 0
                dist_ele.append(dist)
            else:
                print u'轨迹点没有按时间顺序排序'
                break
        result.append(dist_ele)
        if count > 100:
            print 'caculating dist_2_array.....'
            count = 0
    point_list[list_len-1].velocity = 0
    # 距离归一化
    return result


def calculate_partial_dist(dist_2_array, num_limit):
    partial_dist_list = []
    for i in range(len(dist_2_array)):
        if i < len(dist_2_array)-num_limit:
            for j in range(num_limit):
                partial_dist_list.append(dist_2_array[i][j])
        else:
            for j in range(i, len(dist_2_array)):
                partial_dist_list.append(dist_2_array[i][j-i])
    return partial_dist_list

# 原先的距离归一化方法
def normalize_dist_array(result, partial_dist_list):
    max_dist = max(partial_dist_list)
    if max_dist > 500:
        max_dist = 200
    if max_dist != 0:
        for i in range(0, len(result)):
            for j in range(i, len(result)):
                if result[i][j-i] < max_dist:
                    result[i][j-i] /= float(max_dist)
                else:
                    result[i][j-i] = 1


# 距离归一化方法2，直接指定最大距离
def normalize_dist_array2(result, max_dist):
    for i in range(0, len(result)):
        for j in range(i, len(result)):
            if result[i][j - i] < max_dist:
                result[i][j - i] /= float(max_dist)
            else:
                result[i][j - i] = 1


'''
def calculate_optimal_sigma(max_partial_dist):
    sigma_value = []
    d = float(1000)/max_partial_dist
    for sigma in numpy.linspace(0.01, 1, 100):
        sigma_value.append((sigma, abs(math.exp(-1*math.pow(d, 2)/math.pow(sigma, 2))+1/math.sqrt(2)-1)))
    sigma_value.sort(key=lambda x: x[1])
    return sigma_value[0][0]
'''


def calculate_optimal_sigma(dist_2_array, velocity_list, num_limit):
    sigma_value = []
    for sigma in numpy.linspace(0.01, 10, 50):
        potential_list = calculate_potential_value(dist_2_array, velocity_list, sigma, num_limit)
        entropy = calculate__entropy(potential_list)
        sigma_value.append((sigma, entropy))
    sigma_list = []
    entropy_list = []
    for ele in sigma_value:
        sigma_list.append(ele[0])
        entropy_list.append(ele[1])
    pl.plot(sigma_list, entropy_list, label='sigma_entropy')
    pl.show()
    sigma_value.sort(key=lambda x: x[1])
    return sigma_value[0][0]


# 计算potential值时不能把整条轨迹的所有点都考虑进去，否者边界上的聚类点会被淹没掉
def calculate_potential_value(dist_2_array, velocity_list, sigma, velocity_sigma, num_limit):
    length = len(dist_2_array)
    medium_num = num_limit/2

    print 'num_limit = '
    print num_limit
    low_limit = num_limit/2   # 下边
    high_limit = length-num_limit/2   # 上边
    potential_list = []
    for i, dist_list in enumerate(dist_2_array):
        value = 0.0
        if low_limit < i < high_limit:
            value += math.exp((-1)*math.pow(dist_2_array[i][0]/sigma, 2))
            for j in range(1, medium_num+1):
                value += math.exp((-1)*math.pow(dist_2_array[i][j]/sigma, 2))
                value += math.exp((-1)*math.pow(dist_2_array[i-j][j]/sigma, 2))
        if i <= low_limit:
            for j in range(0, num_limit-i):
                value += math.exp((-1)*math.pow(dist_2_array[i][j]/sigma, 2))
            for k in range(i):
                value += math.exp((-1)*math.pow(dist_2_array[k][i-k]/sigma, 2))
        if i >= high_limit:
            for j in range(0, length-i):
                value += math.exp((-1)*math.pow(dist_2_array[i][j]/sigma, 2))
            for k in range(length-num_limit, i):
                value += math.exp((-1)*math.pow(dist_2_array[k][i-k]/sigma, 2))
        value *= math.exp((-1)*math.pow(velocity_list[i] / velocity_sigma, 2))  # 加权
        potential_list.append(value)
    return potential_list


def calculate__entropy(potential_list):
    num = 100
    max_potential = max(potential_list)
    min_potential = min(potential_list)
    step = (max_potential-min_potential)/num
    gap = max_potential-min_potential
    # 归一化
    normalize = [0 for i in range(num+1)]
    for potential in potential_list:
        normalize[int((potential-min_potential)/step)] += 1

    total_num = len(potential_list)
    result = 0
    for ele in normalize:
        if ele>0:
            prob = float(ele)/total_num
            result -= prob*math.log(prob,  2)
    
    return result

# 产生高斯核


def generate_gaus_kernel(sigma):
    result = []
    total = 0
    index = numpy.linspace(-10, 10, 21)
    for i in index:
        ele = math.exp((-1)*i*i/(2*sigma*sigma))/math.sqrt(2*math.pi*sigma*sigma)
        total += ele
        result.append(ele)
    result = [value/total for value in result]

    return result


def calculate_conv(para_list, kernel):

    kernel_len = len(kernel)
    para_len = len(para_list)
    low_limit = kernel_len/2
    high_limit = para_len-low_limit-1
    result = []
    for i in range(para_len):
        value = 0.0
        if low_limit <= i <= high_limit:
            for j in range(i-kernel_len/2, i+kernel_len/2+1):
                value += round(para_list[j]*kernel[kernel_len/2+j-i], 4)
                # print '%f * %f' % (para_list[j], kernel[kernel_len/2+j-i])
            # print 'value is %f' % value
        # 循环卷积
        if i < low_limit:
            for j in range(0, i+kernel_len/2+1):
                value += round(para_list[j]*kernel[kernel_len/2+j-i], 4)
                
            for k in range(0, kernel_len/2-i):
                # value += para_list[para_len-1-k]*kernel[k]
                value += round(para_list[0]*kernel[k], 4)

        if i > high_limit:
            for j in range(i-kernel_len/2, para_len):
                value += round(para_list[j]*kernel[kernel_len/2+j-i], 4)
                # print '%f * %f' % (para_list[j], kernel[kernel_len/2+j-i])
            for k in range(0, i-high_limit):
                value += round(para_list[para_len-1]*kernel[kernel_len/2+para_len-i+k], 4)
                # print '%f * %f' % (para_list[para_len-1], kernel[kernel_len/2+para_len-i+k])

            # print 'value is %f' % value

        result.append(value)
        
    return result


# KD_Curvature
def calculate_curvature(para_list):
    interval = 5
    curvature_list = []
    for i in range(5, len(para_list)-interval):   # 新方法，假定中心点都应该在前半部分/(2/3部分)
        vec1 = [-5]
        vec2 = [5]
        vec1.append(para_list[i-5]-para_list[i])
        vec2.append(para_list[i+5]-para_list[i])
        len1 = math.sqrt(math.pow(vec1[0], 2)+math.pow(vec1[1], 2))
        len2 = math.sqrt(math.pow(vec2[0], 2)+math.pow(vec2[1], 2))
        
        if len1 == 0 or len2 == 0:
            angel = math.pi
        else:
            temp = (vec1[0]*vec2[0]+vec1[1]*vec2[1])/(len1*len2)
            if temp>1 or temp<-1:
                print 'out or range'
                angel = math.pi
            else:
                angel = math.acos(temp)
        value = (math.pi-angel)*2*(len1+len2)
        curvature_list.append(value)
    max_curvature = max(curvature_list)
    result = -100
    for i, ele in enumerate(curvature_list):
        if ele == max_curvature:
            result = i
            break
    result += 5
    return result


# 先归一化在进行KD_curvature
def calculate_curvature3(para_list, is_potential):
    x_list = []
    result_list = []
    max_value = max(para_list)
    curvature_list = []
    for i in range(len(para_list)):
        x_list.append(float(i)/len(para_list))   # 归一化
        result_list.append(para_list[i]/max_value)  # 归一化
    len_limit = 0.1    # 按0.05的曲线长度分段
    last_index = 0
    index_list = [0]
    
    while last_index < len(x_list)-1:
        if is_potential:
            if result_list[last_index] >= 0.001:
                for i in range(last_index, len(x_list)):
                    length = math.sqrt(math.pow(x_list[last_index]-x_list[i], 2) +
                                       math.pow(result_list[last_index]-result_list[i], 2))
                    if i == len(x_list)-1:
                        index_list.append(i)
                        last_index = i
                    if length > len_limit:
                        index_list.append(i)   # index_list=[1,30,68,100,....]
                        last_index = i
                        break
                    else:
                        continue
            else:
                break
        else:   
            for i in range(last_index, len(x_list)):
                length = math.sqrt(math.pow(x_list[last_index]-x_list[i], 2) +
                                   math.pow(result_list[last_index]-result_list[i], 2))
                if i == len(x_list)-1:
                    index_list.append(i)
                    last_index = i
                if length > len_limit:
                    index_list.append(i)
                    last_index = i
                    break
                else:
                    continue
    print '*'*30
    print index_list

    # 绘制采样后的点
    # print u'绘制采样后的点'
    # x_after = []
    # result_after = []
    # for index in index_list:            # index_list=[1,30,68,100,....]
    #     x_after.append(x_list[index])
    #     result_after.append(result_list[index])
    # pl.figure(figsize=figure_size)
    # l1 = pl.plot(x_after, result_after, 'og')
    # pl.setp(l1, markersize=4)
    # pl.xlabel('normalized sequence number')
    # if is_potential:
    #     pl.ylabel('normalized density')
    # else:
    #     pl.ylabel('normalized distance')
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\sample density.png', dpi=200)
    # pl.show()
    # *************************************************
    #  做剔除处理，若是密度序列只考虑密度大于0.7的那部分
    partial_index_list = []
    if is_potential:
        for index in index_list:
            if result_list[index] >= 0:
                partial_index_list.append(index)
    else:
        partial_index_list = index_list
    # *************************************************
    for i in range(1, len(partial_index_list)-1):
        vec1 = []
        vec2 = []
        vec1.append(x_list[partial_index_list[i-1]] - x_list[partial_index_list[i]])
        vec2.append(x_list[partial_index_list[i+1]] - x_list[partial_index_list[i]])
        vec1.append(result_list[partial_index_list[i-1]] - result_list[partial_index_list[i]])
        vec2.append(result_list[partial_index_list[i+1]] - result_list[partial_index_list[i]])
                  
        len1 = math.sqrt(math.pow(vec1[0], 2)+math.pow(vec1[1], 2))
        len2 = math.sqrt(math.pow(vec2[0], 2)+math.pow(vec2[1], 2))
        if len1 == 0 or len2 == 0:
            angel = math.pi
        else:
            temp = (vec1[0]*vec2[0]+vec1[1]*vec2[1])/(len1*len2)
            if temp > 1 or temp < -1:
                print 'out or range'
                angel = math.pi
            else:
                angel = math.acos(temp)
        value = (math.pi-angel)*2*(len1+len2)
        curvature_list.append(value)
    max_curvature = max(curvature_list)
    print u'曲率值'
    print curvature_list
    print 'max'
    print max_curvature
    result = -100
    for i, ele in enumerate(curvature_list):
        if ele == max_curvature:
            result = i            # result = i+1
            break
    '''
    index1 = partial_index_list[result]
    index2 = partial_index_list[result+1]
    result = int((index1+index2)/2)
    '''
    result = partial_index_list[result+1]

    # 画出曲率图（归一化后的）

    # fig = pl.figure(figsize=figure_size)
    # ax = fig.add_subplot(111, autoscale_on=False)
    # ax.plot(x_list, result_list, 'g', linewidth=line_width)
    # elbow_point_mark = ax.plot(x_list[result], result_list[result], '*r')
    # pl.setp(elbow_point_mark, markersize=marker_size)
    # pl.setp(elbow_point_mark, markerfacecolor=marker_face_color)
    # ax.annotate('elbow point', xy=(x_list[result], result_list[result]), xycoords='data',
    #             xytext=(x_list[result]+0.2, result_list[result]+0.2), textcoords='data',
    #             # arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, frac=0.1),
    #             arrowprops=dict(arrowstyle="->"),
    #             horizontalalignment='right', verticalalignment='top',
    #             )
    # pl.xlabel('normalized sequence number')
    # pl.ylabel('normalized density')
    # pl.legend()
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\elbow point.png', dpi=200)
    # pl.show()
    # 画局部放大图
    # plt.figure()
    # p_local = plt.subplot(111)
    # p_local.axis([0.64, 0.72, 0.95, 0.99])
    # p_local.plot(x_list, result_list, label='normalized')
    # p_local.plot(x_list[result], result_list[result], '*r', linewidth=30)
    # p_local.set_xlabel('index')
    # p_local.set_ylabel('density')
    # plt.show()
    return result


def calculate_curvature2(para_list):  # 按曲线的二次导数判断
    interval_list = []
    
    print 'para_list'
    print para_list
    
    for i in range(0, len(para_list)-1):
        if para_list[i]-para_list[i+1] == 0:
            continue
        else:
            interval_list.append(para_list[i]-para_list[i+1])
    max_interval = max(interval_list)
    
    print 'interval_list'
    print interval_list
    
    index = 0
    for i, interval in enumerate(interval_list):
        if max_interval == interval:
            index = i
            break
    rate_list = []
    for k in range(index, len(interval_list)-1):  # 从最大值开始算
        if interval_list[k] > 0.01:
            rate_list.append(interval_list[k]/interval_list[k+1])
        
    print 'rate_list'
    print rate_list
    
    max_rate = max(rate_list)
    result = -1
    for i, rate in enumerate(rate_list):
        if rate == max_rate:
            result = i+index
            break
    result += 2
    print 'I am here '
    print 'index %d' % index
    print 'final_index %d' % result
    return result
        

def test():
    result = []
    total = 0
    index = numpy.linspace(-1, 1, 201)
    for i in index:
        ele = math.exp((-1)*i*i)
        total += ele
        result.append(ele)
    pl.plot(result, 'ob')
    pl.show()

#-----------------------下面函数是专门用于New_DBSCAN的---------------------


def caculate_adjacent_dist_list(points, num_limit):
    distance_list = []
    total_len = len(points)
    for idx, point in enumerate(points):
        if idx < num_limit/2:
            count = 1
            ele_list = []
            while count <= num_limit:
                dist = distance_calculate(points[idx], points[count-1])
                ele_list.append(dist)
                count += 1
            distance_list.append(ele_list)
        else:
            if idx >= len(points) - num_limit/2:
                count = 1
                ele_list = []
                while count <= num_limit:
                    dist = distance_calculate(points[idx], points[total_len-count])
                    ele_list.append(dist)
                    count += 1
                distance_list.append(ele_list)
            else:
                from_idx = idx - num_limit/2
                end_idx = idx + num_limit/2
                ele_list = []
                while from_idx <= end_idx:
                    dist = distance_calculate(points[idx], points[from_idx])
                    ele_list.append(dist)
                    from_idx += 1
                distance_list.append(ele_list)
    return distance_list


def normalize_adjacent_dist_list(distance_list):
    max_distance_list = [max(dist_list) for dist_list in distance_list]
    max_distance = max(max_distance_list)
    ele_len = len(distance_list[0])
    print 'ele_len=' + str(ele_len)
    for i in range(len(distance_list)):
        if ele_len != len(distance_list[i]):
            print i
            print 'break! '*10
        for j in range(len(distance_list[i])):
            # distance_list[i][j] /= float(max_distance)
            if distance_list[i][j] > 200:   #直接把部分距离的上限设为200米
                distance_list[i][j] = 1
            else:
                distance_list[i][j] /= float(200)


# 计算potential值时不能把整条轨迹的所有点都考虑进去，否者边界上的聚类点会被淹没掉
def calculate_density_value(distance_list, velocity_list, sigma, velocity_sigma):
    potential_list = []
    for i, dist_list in enumerate(distance_list):
        value = 0.0
        for j in range(len(dist_list)):
            value += math.exp((-1) * math.pow(dist_list[j] / sigma, 2))
        value *= math.exp((-1)*math.pow(velocity_list[i] / velocity_sigma, 2))  # 加权
        potential_list.append(value)
    return potential_list


def caculate_velocity(points):
    for i, point in enumerate(points):
        if i < len(points) - 1 and points[i+1].time > point.time:
            points[i].velocity = distance_calculate(point, points[i+1]) / \
                                 float((points[i+1].time - points[i].time) / float(1000))
        else:
            point.velocity = 0

# 计算所有相邻轨迹点之间的时间间隔的平均值
def caculate_mean_interval_time(points):
    interval_time_list = []
    num = len(points)
    mean_interval = 0
    for i in range(len(points)-1):
        interval = points[i+1].time - points[i].time
        interval_time_list.append(interval)
        mean_interval += interval / float(num)
    pl.plot(interval_time_list)
    pl.show()
    return mean_interval


# 聚类合并，将小聚类合并成大聚类
def merge_clusters(clusters):
    new_clusters = []
    if len(clusters) == 0:
        return new_clusters
    else:
        prev_cluster = clusters[0]
        for i in range(1, len(clusters)):
            prev_centre_point = calculate_cluster_centre(prev_cluster)
            next_cluster = clusters[i]
            next_centre_point = calculate_cluster_centre(next_cluster)
            prev_cluster_end_time = max([point.time for point in prev_cluster])
            next_cluster_begin_time = min([point.time for point in next_cluster])
            centre_distance = distance_calculate(prev_centre_point, next_centre_point)
            time_interval = next_cluster_begin_time - prev_cluster_end_time
            print '*' * 10
            print 'centre_distance=' + str(centre_distance)
            print 'time_interval=' + str(time_interval/1000/60)
            if centre_distance <= 200 and time_interval <= 60*60*1000:
                for point in next_cluster:
                    prev_cluster.append(point)
            else:
                new_clusters.append(prev_cluster)
                prev_cluster = next_cluster
        new_clusters.append(prev_cluster)      # 最后一个聚类需要额外加进去
        return new_clusters


# 对停留点进行提纯，根据停留时间的长短
def clusters_refinement(clusters):
    new_clusters = []
    if len(clusters) == 0:
        return new_clusters
    else:
        for cluster in clusters:
            max_time = max([point.time for point in cluster])
            min_time = min([point.time for point in cluster])
            duration = (max_time - min_time) / 1000
            if duration > 2 * 60:
                new_clusters.append(cluster)
            else:
                continue
    return new_clusters



def calculate_cluster_centre(cluster):
    lon = 0.0
    lat = 0.0
    total_len = len(cluster)
    for point in cluster:
        lon += point.lon/total_len
        lat += point.lat/total_len
    centre_point = TrackPoint(lon, lat, 0)
    return centre_point

# 显示停留点
def show_clusters(clusters):
    colors = ['or', 'ob', 'og', 'oy', 'ok', 'oc']
    color_idx = 0
    for points in clusters:
        lon = [point.lon for point in points]
        lat = [point.lat for point in points]
        pl.plot(lon, lat, colors[color_idx % len(colors)])
        color_idx += 1
    pl.show()


# 在轨迹上显示停留点
def show_clusters_trajectory(clusters, original_track):
    x = [point.lon for point in original_track]
    y = [point.lat for point in original_track]
    pl.plot(x, y, 'g')
    colors = ['or', 'ob', 'og', 'oy', 'ok', 'oc']
    color_idx = 0
    for points in clusters:
        lon = [point.lon for point in points]
        lat = [point.lat for point in points]
        pl.plot(lon, lat, colors[color_idx % len(colors)])
        color_idx += 1
    pl.show()



def print_clusters_info(clusters):
    print "total clusters num=" + str(len(clusters))
    for points in clusters:
        print '-'*10
        time_list = [point.time for point in points]
        print 'min_time=' + str(min(time_list))
        print 'max_time=' + str(max(time_list))
        print 'point_number=' + str(len(points))
        print 'duration=' + str((max(time_list)-min(time_list))/1000/60) + 'minutes'
        print 'lon=' + str(points[0].lon) + 'lat=' + str(points[0].lat)


# ------------------------------------------------------------------------------------

# 尝试用距离的比值看看
def distance_rate(file_name):
    show_track.show_track(file_name)
    data = read_geolife_data_file(file_name)
    # data = read_own_data_file(file_name)
    compressed_data = data_preprocessing(data)
    # 相邻轨迹点之间的距离序列
    distance_list = []
    for i in range(len(compressed_data)-1):
        d = distance_calculate(compressed_data[i], compressed_data[i+1])
        distance_list.append(d)
    pl.plot(distance_list)
    pl.show()
    num_gap = 71
    length = len(compressed_data)
    stability_list = []
    for i in range(len(compressed_data)):
        index_pair = get_index_pair(i, length, num_gap)
        continue_dist = 0
        for j in range(index_pair[0], index_pair[1]):
            continue_dist += distance_list[j]  #末尾特殊情况用前一个值去代替
        direct_dist = distance_calculate(compressed_data[index_pair[0]], compressed_data[index_pair[1]])
        if continue_dist != 0:
            stability_list.append(float(direct_dist)/continue_dist)
        else:
            stability_list.append(stability_list[len(stability_list)-1])  #若分母为0，则认为与上一个特征相同
    pl.plot(stability_list)
    pl.xlabel('sequence')
    pl.ylabel('move ability')
    # pl.title('stability')
    pl.show()
    kernel = generate_gaus_kernel(4)
    stability_list = calculate_conv(stability_list, kernel)
    pl.plot(stability_list)
    pl.title('smoothed stability')
    pl.show()
    # 备份
    temp = []
    for s in stability_list:
        temp.append(s)
    temp.sort()
    pl.plot(temp)
    pl.title('sorted stability')
    pl.show()
    clusters = []
    cluster_idx = []
    is_first = True
    for i in range(len(stability_list)):
        if stability_list[i] > 0.4:
            if is_first:
                continue
            else:
                cluster = []
                if len(cluster_idx) > 0:
                    for idx in cluster_idx:
                        cluster.append(compressed_data[idx])
                    clusters.append(cluster)
                cluster_idx = []
        else:
            cluster_idx.append(i)
            is_first = False
        if i == len(stability_list)-1 and stability_list[i] < 0.4:
            cluster = []
            for idx in cluster_idx:
                cluster.append(compressed_data[idx])
            clusters.append(cluster)
    print 'total_cluster:' + str(len(clusters))
    print_clusters_info(clusters)
    show_clusters_trajectory(clusters, compressed_data)


def calculate_stability(compressed_data, num_gap):
    distance_list = []
    for i in range(len(compressed_data) - 1):
        d = distance_calculate(compressed_data[i], compressed_data[i + 1])
        distance_list.append(d)
    length = len(compressed_data)
    stability_list = []
    for i in range(len(compressed_data)):
        index_pair = get_index_pair(i, length, num_gap)
        continue_dist = 0
        for j in range(index_pair[0], index_pair[1]):
            continue_dist += distance_list[j]  # 末尾特殊情况用前一个值去代替
        direct_dist = distance_calculate(compressed_data[index_pair[0]], compressed_data[index_pair[1]])
        if continue_dist != 0:
            stability_list.append(float(direct_dist) / continue_dist)
        else:
            stability_list.append(stability_list[len(stability_list) - 1])  # 若分母为0，则认为与上一个特征相同
    # pl.plot(stability_list)
    # pl.title('stability')
    # pl.show()
    kernel = generate_gaus_kernel(4)
    stability_list = calculate_conv(stability_list, kernel)
    # pl.plot(stability_list)
    # # pl.title('smoothed stability')
    # pl.xlabel('sequence number')
    # pl.ylabel('smoothed move ability')
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\1\\smoothed  move ability.png', dpi=600)
    # pl.show()
    stability_list.append(stability_list[len(stability_list)-1])  #末尾的特殊处理，增加一个特征值
    return stability_list


def get_index_pair(idx, length, num_gap):
    result = []
    if idx <= num_gap/2:
        result.append(0)
        result.append(num_gap-1)
    else:
        if idx >= length-num_gap/2:
            result.append(length-num_gap)
            result.append(length-1)
        else:
            result.append(idx-num_gap/2)
            result.append(idx+num_gap/2)
    return result


# 查看特征值的分布
def show_value_distribution(feature_list):
    total_length = len(feature_list)
    value_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    value_count = [0 for i in range(len(value_list))]
    for value in feature_list:
        if value == 1:
            value_count[9] += 1
        else:
            idx = int(value / 0.1)
            value_count[idx] += 1
    for i in range(len(value_count)):
        value_count[i] = round(1.0 * value_count[i] / total_length,4)
    value_count_tuple = tuple(value_count)
    value_list_tuple = tuple(value_list)
    rects = plt.bar(value_list_tuple, value_count_tuple, width=0.1, align="edge", yerr=0.000001)
    plt.xlabel('normalized velocity')
    plt.ylabel('percentage')
    plt.show()


# 计算聚类结果的SSE稀疏
def calculate_sse_coeff(clusters):
    sse_sum = 0
    for cluster in clusters:
        points_len = len(cluster)
        lon_sum = 0.0
        lat_sum = 0.0
        for point in cluster:
            lon_sum += point.lon
            lat_sum += point.lat
        lon_mean = 1.0 * lon_sum / points_len
        lat_mean = 1.0 * lat_sum / points_len
        center_point = TrackPoint(lon_mean, lat_mean, 0)
        dist_sum = 0
        for point in cluster:
            dist_sum += distance_calculate(point, center_point)
        dist_mean = 1.0 * dist_sum / len(cluster)
        sse_sum += dist_mean
    sse = 1.0 * sse_sum / len(clusters)
    return sse




