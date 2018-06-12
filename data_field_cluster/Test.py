# _*_coding:utf-8_*_ #

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import math
import Utility
from TrackPoint import TrackPoint

figure_size = (8, 6)


# def f(t):
#     s1 = np.cos(2 * np.pi * t)
#     e1 = np.exp(-t)
#     return s1 * e1
# t1 = np.arange(0.0, 5.0, .2)
# fig = pl.figure(figsize=figure_size, dpi=100, facecolor='white')
# l = pl.plot(t1, f(t1), 'r', label='china', linewidth=2)
# pl.setp(l, 'markersize', 10)
# pl.setp(l, 'markerfacecolor', 'b')
# pl.xlabel('xxx')
# pl.ylabel('yyy')
# pl.legend()
# pl.show()
# ***************************************************** #

# if 1:
# # if only one location is given, the text and xypoint being
# # annotated are assumed to be the same
#     fig = plt.figure()
#     ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 5), ylim=(-3, 5))
#     t = np.arange(0.0, 5.0, 0.01)
#     s = np.cos(2*np.pi*t)
#     line, = ax.plot(t, s, lw=3, color='purple')
#     ax.annotate('axes center', xy=(.5, .5), xycoords='axes fraction',
#                 horizontalalignment='center', verticalalignment='center')
#     ax.annotate('pixels', xy=(20, 20), xycoords='figure pixels')
#     ax.annotate('points', xy=(100, 300), xycoords='figure points')
#     ax.annotate('offset', xy=(1, 1), xycoords='data',
#                 xytext=(-15, 10), textcoords='offset points',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='right', verticalalignment='bottom',)
#     # ax.annotate('local max', xy=(3, 1), xycoords='data',
#     #             xytext=(0.8, 0.95), textcoords='axes fraction',
#     #             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6,),
#     #             horizontalalignment='right', verticalalignment='top',
#     #             )
#     ax.annotate('local max', xy=(3, 1), xycoords='data',
#                 xytext=(0.8, 0.95), textcoords='axes fraction',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='right', verticalalignment='top',
#                 )
#     ax.annotate('a fractional title', xy=(.025, .975),
#                 xycoords='figure fraction',
#                 horizontalalignment='left', verticalalignment='top', fontsize=20)
# # points above the bottom
#     ax.annotate('bottom right (points)', xy=(-10, 10), xycoords='axes points',
#                 horizontalalignment='right', verticalalignment='bottom', fontsize=20)
# if 2:
#     fig2 = plt.figure(figsize=(8, 6))
#     axes2 = fig2.add_subplot(111)
#     x = np.arange(0, 4, 0.05)
#     y = np.sin(x)
#     l = axes2.plot(x, y, label='sin()')
#     axes2.annotate('max value', xy=(np.pi/2, 1), xycoords='data',
#                    xytext=(np.pi/2-0.2, 0.7), textcoords='data',
#                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6,),
#                    horizontalalignment='left', verticalalignment='centre')
#     plt.setp(l, markersize=20)
# plt.show()

# x = np.arange(0, 1, 0.01)
# y = [1/(1 + math.pow(math.e, -(10*ele-5))) for ele in x]
# plt.plot(x, y)
# plt.xlabel('t')
# plt.ylabel('temporal distance')

# plt.show()





    # x = [ele.split(' | ')[3] for ele in open(file_name1, 'r')]
    # y = [ele.split(' | ')[4] for ele in open(file_name1, 'r')]
    # plt.plot(x, y, 'g')
    # x2 = [ele.split(' ')[0] for ele in open(file_name2, 'r')]
    # y2 = [ele.split(' ')[1] for ele in open(file_name2, 'r')]
    # plt.plot(x, y, 'og')
    # plt.plot(x2, y2, 'or')
    # plt.show()
    # compressed_data = Utility.read_geolife_data_file(file_name1)
    # clustered_data = Utility.read_geolife_data_file(file_name2)
    # x = []
    # y = []
    # for point in compressed_data:
    #     if point not in clustered_data:
    #         x.append(point.lon)
    #         y.append(point.lat)
    # plt.plot(x, y, 'or')
    # plt.show()

p1 = TrackPoint(116.319857, 39.987436, 20000)
p2 = TrackPoint(116.319872, 39.987406, 22000)
dist = Utility.distance_calculate(p1, p2)
print dist

