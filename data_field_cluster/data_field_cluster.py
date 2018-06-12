#  _*_coding:utf-8_*_
import Utility
import math
import pylab as pl
import matplotlib
from TrackPoint import TrackPoint


def calculate_optimal_potential(dist_2_array, velocity_list, num_limit):
    sigma = 0.01
    entropy_list = []
    sigma_list = []
    while sigma <= 1:
        potential_list = Utility.calculate_potential_value(dist_2_array, velocity_list, sigma, num_limit)
        entropy = Utility.calculate__entropy(potential_list)
        entropy_list.append(entropy)
        sigma_list.append(sigma)
        print '*'*10+str(sigma)+'*'*10
        sigma += 0.05
    # min_entropy = min(entropy_list)
    # 选出entropy最小时对应的index
    min_entropy = max(entropy_list)

    print '最小entropy：'+str(min_entropy)
    index = 0
    for i, entropy in enumerate(entropy_list):
        if entropy == min_entropy:
            index = i
            break
    potential_list = Utility.calculate_potential_value(dist_2_array, velocity_list, sigma_list[index], num_limit)
    print '*'*10+'the optimal sigma is'+str(sigma_list[index])
    print len(sigma_list) == len(entropy_list)
    pl.plot(sigma_list, entropy_list, 'b', label = 'sigma_entropy')
    pl.xlabel('sigma')
    pl.ylabel('potential_entropy')
    pl.legend()
    pl.show()

    return potential_list

# 速度归一化方法（1）
def normalize_point_velocity(point_list):
    # 速度在此进行了平滑
    vec_list = [point.velocity for point in point_list]
    pl.plot(vec_list)
    pl.title('velocity without smooth')
    pl.show()
    kernel = Utility.generate_gaus_kernel(4)
    vec_list = Utility.calculate_conv(vec_list, kernel)
    pl.plot(vec_list)
    pl.title('vecolity after smooth')
    pl.show()
    max_vec = max(vec_list)
    min_vec = min(vec_list)
    print 'max_vec is %f' % max_vec
    print 'min_vec is %f' % min_vec
    gap = max_vec-min_vec
    if gap > 0:
        for i in range(len(vec_list)):
            interval = vec_list[i]-min_vec
            point_list[i].velocity = round(interval/gap, 4)
    '''       
    f = open('velocity2.txt', 'w')
    for i in range(len(point_list)):
        # print 'velocity is %f' % point_list[i].velocity
        f.write('velocity is %f\n' % point_list[i].velocity)
    f.close()
    '''

# 速度归一化方法（2）
def normalize_velocity(velocity_list):
    # pl.plot(velocity_list)
    # pl.title('velocity before normalization')
    # pl.show()

    # max_vec = max(velocity_list)
    max_vec = min(10, max(velocity_list))
    min_vec = min(velocity_list)
    gap = max_vec - min_vec
    weight_list = []
    if gap > 0:
        for i in range(len(velocity_list)):
            interval = velocity_list[i] - min_vec
            # interval = velocity_list[i] - min_vec
            velocity_list[i] = min(1, round(interval / gap, 4))
            weight_list.append(math.exp(-1*math.pow(velocity_list[i]/0.5, 2)))
    # pl.plot(weight_list)
    # pl.title('weight list')
    # pl.show()

    return velocity_list


# 局部速度序列速度归一化方法
def normalize_partial_vec_list(partial_vec_list):
    max_list = []
    min_list = []
    for vec_list in partial_vec_list:
        max_ele = max(vec_list)
        min_ele = min(vec_list)
        max_list.append(max_ele)
        min_list.append(min_ele)
    max_vec = max(5, max(max_list))
    min_vec = min(min_list)
    gap = max_vec - min_vec
    if gap > 0:
        for i in range(len(partial_vec_list)):
            for j in range(len(partial_vec_list[i])):
                interval = partial_vec_list[i][j] - min_vec
                partial_vec_list[i][j] = min(1, round(interval / gap), 4)
    return partial_vec_list



# 密度归一化方法
def normalize_density(density_list):
    max_den = max(density_list)
    min_den = min(density_list)
    gap = max_den - min_den
    if gap > 0:
        for i in range(len(density_list)):
            interval = density_list[i] - min_den
            density_list[i] = round(interval / gap, 4)
    return density_list

def normalize_use_score(point_list):
    vec_list = [point.velocity for point in point_list]
    sum_vec = sum(vec_list)
    mean_vec = sum_vec/len(point_list)
    variance = 0
    for i in range(len(point_list)):
        variance += math.pow(point_list[i].velocity-mean_vec, 2)
    variance /= len(point_list)
    std_variance = math.sqrt(variance)
    for j in range(len(point_list)):
        point_list[j].velocity = (vec_list[j]-mean_vec)/std_variance

    f = open('velocity.txt', 'w')
    for i in range(len(point_list)):
        # print 'velocity is %f' % point_list[i].velocity
        f.write('velocity is %f\n' % point_list[i].velocity)
        pl.plot(i, point_list[i].velocity, 'ob')
    f.close()
    pl.show()


def calculate_potential_threshold(potential_list):
    frame_x = [110, 180, 180, 110, 110]
    frame_y = [22.5, 22.5, 28, 28, 22.5]
    potential_list.sort(reverse=True)
    # 画平滑前的密度序列图（未归一化）
    # pl.figure(figsize=Utility.figure_size)
    # l1 = pl.plot(potential_list, 'og')
    # pl.plot(frame_x, frame_y, 'k')
    # pl.xlabel('sequence number')
    # pl.ylabel('density')
    # pl.setp(l1, markersize=3)
    # # # 画局部
    # a = pl.axes([0.44, 0.4, 0.44, 0.48], axisbg='w')
    # x = range(110, 180)
    # y = [potential_list[i] for i in x]
    # l2 = pl.plot(x, y, 'og')
    # pl.xlabel('sequence number')
    # pl.ylabel('density')
    # pl.setp(l2, markersize=3)
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\density before smooth.png', dpi=200)
    # pl.show()

    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # pl.plot(potential_list, 'og')
    # fig.savefig('test2png.png', dpi=1000)


    max_potential = max(potential_list)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_potential_list = Utility.calculate_conv(potential_list, kernel)
    # 画平滑后的密度序列图
    # pl.figure(figsize=Utility.figure_size)
    # l3 = pl.plot(smoothed_potential_list, 'og')
    # pl.plot(frame_x, frame_y, 'k')
    # pl.xlabel('sequence number')
    # pl.ylabel('smoothed density')
    # pl.setp(l3, markersize=3)
    # a = pl.axes([0.44, 0.4, 0.44, 0.48], axisbg='w')
    # x = range(110, 180)
    # y = [smoothed_potential_list[i] for i in x]
    # l4 = pl.plot(x, y, 'og')
    # pl.xlabel('sequence number')
    # pl.ylabel('smoothed density')
    # pl.setp(l4, markersize=3)
    # pl.savefig('C:\\Users\\Administrator\\Desktop\\Figure 1\\density after smooth.png', dpi=200)
    # pl.title('sorted density sequence after smooth')
    # pl.show()
    curvature_index = Utility.calculate_curvature3(smoothed_potential_list, True)

    if curvature_index > 0:
        threshold_candid = smoothed_potential_list[curvature_index]
        if max_potential/threshold_candid > 2:
            print '*'*30
            print max_potential
            print threshold_candid
            print '*'*30
            threshold_candid = max_potential/math.sqrt(2)
        threshold = threshold_candid
        return threshold
    else:
        return -1


def calculate_dist_threshold(dist_list):
    dist_list.sort(reverse=True)
    # max_dist = max(dist_list)
    kernel = Utility.generate_gaus_kernel(4)
    smoothed_dist_list = Utility.calculate_conv(dist_list, kernel)
    curvature_index = Utility.calculate_curvature3(smoothed_dist_list, False)

    print '返回值'
    print curvature_index
    print smoothed_dist_list[curvature_index]

    pl.plot(smoothed_dist_list, 'ob', label='smoothed_dist')
    pl.plot(dist_list, 'og', label='dist_statistic')
    pl.xlabel('sequence')
    pl.ylabel('potential_value')
    pl.legend()
    pl.show()

    if curvature_index > 0:
        '''
        threshold_candid = dist_list[curvature_index]
        gap = max_dist-threshold_candid
        threshold1 = threshold_candid+gap*(1-1/math.sqrt(2))
        
        threshold2 = dist_list[curvature_index]
        if threshold1>threshold2:
            threshold = threshold1
        else:
            threshold = threshold2
        '''
        threshold = smoothed_dist_list[curvature_index]
        return threshold
    else:
        return -1


def calculate_density_distance(dist_2_array, potential_list):
    index_potential = []
    for i, potential in enumerate(potential_list):
        index_potential.append((i, potential))
    index_potential.sort(key=lambda x: x[1], reverse=True)  # 按降序排序
    length = len(index_potential)
    result = [0 for ele in range(length)]  # 初始化
    for i in range(length):
        potential = index_potential[i]
        index = potential[0]       
        if i == 0:
            result[index] = 0
            continue
        index_list = [ele[0] for ele in index_potential[0:i]]   
        min1 = 1000000
        for k in index_list:            # 取出索引号
            if index > k:
                if dist_2_array[k][index-k] < min1:
                    min1 = dist_2_array[k][index-k]
            if k > index:
                if dist_2_array[index][k-index] < min1:
                    min1 = dist_2_array[index][k-index]
        result[index] = min1
    # 为避免最大值干扰，把最大值做一定处理
    dist = max(result)
    result[index_potential[0][0]] = dist
    return result


def add_attributes(data, potential_list, dist_list):  # 把计算得到的potential和dist属性赋给点数据
    if len(potential_list) == len(dist_list):
        for i in range(len(potential_list)):
            data[i].potential = potential_list[i]
            data[i].distance = dist_list[i]
    else:
        print 'potential_list and dist_list,  the length is not the same'


# 根据得到的聚类中心找到聚类
def get_cluster(centre_index_list, data, dist_2_array, potential_threshold, dist_threshold):
    cluster_list = []
    for i in range(len(centre_index_list)):
        cluster = []
        cluster_list.append(cluster)
    for i in range(len(data)):
        point = data[i]
        if point.potential>potential_threshold and point.distance<dist_threshold:
            dist_list = []
            for index in centre_index_list:
                if i>= index:
                    dist_list.append(dist_2_array[index][i-index])
                else:
                    dist_list.append(dist_2_array[i][index-i])
            min_dist = min(dist_list)
            for k in range(len(dist_list)):
                if dist_list[k] == min_dist:
                    cluster_list[k].append(point)
                    break
    return cluster_list


def result_improvement(compressed_data, num_limit, centre_index_list):
    result = []
    for index in centre_index_list:
        print'-'*30
        count1 = 0
        count2 = 0
        last_sym = True
        flag = True
        low = 0
        high = 0
        if num_limit/2-1 < index < len(compressed_data)-num_limit/2-1:
            low = index-num_limit/2
            high = index+num_limit/2+1
        if index <= num_limit/2-1:
            low = 0
            high = num_limit-1
        if index >= len(compressed_data)-num_limit/2-1:
            low = len(compressed_data)-num_limit
            high = len(compressed_data)-1
            
        for i in range(low, high):
            if compressed_data[i].lon == compressed_data[i+1].lon and compressed_data[i].lat == compressed_data[i+1].lat:
                count1 += 1
                count2 += 1
                print '*'*20+'sa'
            else:
                vec = [compressed_data[i+1].lon-compressed_data[i].lon, compressed_data[i+1].lat-compressed_data[i].lat]
                module = math.sqrt(math.pow(vec[0], 2)+math.pow(vec[1], 2))
                cos = (vec[0]*1+vec[1]*0)/float(module)
                if cos>0:
                    flag = True
                if cos<0:
                    flag = False
                if flag^last_sym:
                    last_sym = flag
                    count1 += 1
                    temp = count1
                    count1 = count2
                    count2 = temp
                else:
                    count2 += 1
        # dir_rate = 0
        
        if count1!= 0 and count2!= 0:
            dir_rate = max([float(count1)/count2, float(count2)/count1])
            print dir_rate
            if dir_rate<5:     # 自己设的一个比值
                result.append(index)
        else:
            continue
        
    return result


def result_show(original_data, stop_point_list):
    x = [point.lon for point in original_data]
    y = [point.lat for point in original_data]
    pl.figure(figsize=Utility.figure_size)
    pl.plot(x, y, 'g', linewidth=Utility.line_width)
    stop_lon = [point.lon for point in stop_point_list]
    stop_lat = [point.lat for point in stop_point_list]
    hotspot_mark = pl.plot(stop_lon, stop_lat, '*r')  # 对热点进行特别标记
    pl.setp(hotspot_mark, markersize=Utility.marker_size)
    pl.setp(hotspot_mark, markerfacecolor=Utility.marker_face_color)
    pl.xlabel('longitude')
    pl.ylabel('latitude')
    pl.legend()
    pl.show()


def refresh_dist_array(point_list, dist_array):
    list_len = len(point_list)
    max_time_interval = point_list[list_len-1].time-point_list[0].time
    for i in range(0, list_len):
        for j in range(i, list_len):
            spatial_dist = dist_array[i][j-i]
            temp_dist = 1/(1+math.pow(math.e, -(10*(point_list[j].time-point_list[i].time)/max_time_interval-5)))
            dist_array[i][j-i] = math.sqrt(math.pow(spatial_dist, 2)+math.pow(temp_dist, 2))


def get_stop_position(compressed_data, centre_index_list):
    result = []
    for index in centre_index_list:
        result.append(TrackPoint(compressed_data[index].lon, compressed_data[index].lat, compressed_data[index].time))
    return result


def merge_stop_position(dist_2_array_copy, centre_index_list, potential_list):
    result = []
    if len(centre_index_list) > 1:
        last_stop_index = centre_index_list[0]
        i = 1
        merge_point = [(last_stop_index, potential_list[last_stop_index])]
        while i < len(centre_index_list):
            next_stop_index = centre_index_list[i]
            is_one = True                            #标注是否是同一个停留点
            for dist in dist_2_array_copy[last_stop_index][0:next_stop_index-last_stop_index]:
                if dist > 200:
                    is_one = False
            if is_one:
                merge_point.append((centre_index_list[i], potential_list[i]))
                print '+'*5
                if i >= len(centre_index_list) - 1:
                    # result.append(next_stop_index)
                    result.append(max(merge_point, key=lambda x: x[1])[0])  # 合并停留点时最终记录的是密度最大的哪个点
                i += 1
            else:
                print '-'*5
                result.append(max(merge_point, key=lambda x: x[1])[0])
                # result.append(last_stop_index)
                last_stop_index = next_stop_index
                merge_point = [(last_stop_index, potential_list[last_stop_index])]
                if i >= len(centre_index_list)-1:
                    result.append(next_stop_index)
                    break
                i += 1
    else:
        return centre_index_list
    return result


def refine_stop_position(dist_2_array_copy, centre_index_list, compressed_data):
    result = []
    dist_limit = 200
    time_limit = 30*60*1000
    for centre_index in centre_index_list:
        low_index = centre_index
        high_index = centre_index
        while True:
            if dist_2_array_copy[low_index][centre_index-low_index] < dist_limit:
                if low_index > 0:
                    low_index -= 1
                    continue
                else:
                    break
            else:
                break
        while True:
            if dist_2_array_copy[centre_index][high_index-centre_index] < dist_limit:
                if high_index < len(compressed_data)-1:
                    high_index += 1
                    continue
                else:
                    break
            else:
                break
        time_interval = compressed_data[high_index].time - compressed_data[low_index].time
        if time_interval > time_limit:
            result.append(centre_index)
        else:
            continue
    return result


# 直接根据密度特征得到聚类
def get_cluster_points(points, density_threshold):
    cluster_points = []
    for point in points:
        if(point.potential >= density_threshold):
            cluster_points.append(point)
    return cluster_points



            
    

    

















