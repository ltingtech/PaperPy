#  _*_coding:utf-8_*_

import math
import matplotlib.pyplot as plt
import pylab as pl
from TrackPoint import TrackPoint
import numpy
import time
import datetime



def show_gaussi(sigma_list):
    for sigma in sigma_list:
        xx = []
        yy = []
        for i in numpy.linspace(0, 1, 200):
            xx.append(i)
            yy.append(math.exp(-1*(i*i)/math.pow(sigma, 2)))
        plt.plot(xx, yy)
    plt.show()


        

# sigma_list = numpy.linspace(0.1, 1, 10)
sigma_list = [0.3]
show_gaussi(sigma_list)
