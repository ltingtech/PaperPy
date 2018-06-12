# scoding:utf-8
import matplotlib.pyplot as pl
import numpy
import math
import Utility


def plot_velocity_weight(velocity_sigma):
    velocity = []
    weight = []
    for ele in numpy.linspace(-1, 1, 200):
        velocity.append(ele)
        weight.append(math.exp((-1)*math.pow(ele / velocity_sigma, 2)))
    pl.figure(figsize=Utility.figure_size)
    l = pl.plot(velocity, weight, 'b')
    pl.setp(l, markersize=3)
    # pl.text(0.4, 0.5, r'$\mu=100,\ \sigma=15$')
    pl.text(0.32, 0.7, r'$\sigma=$'+str(velocity_sigma))
    pl.title('weight function')
    pl.xlabel('velocity')
    pl.ylabel('weight')
    pl.show()


def plot_velocity_weight_mult(velocity_sigma_begin, velocity_sigma_end):
    colors = ['r', 'b', 'g', 'y', 'k', 'm']
    # first_sigma = velocity_sigma_begin
    # sigma_interval = (velocity_sigma_end - velocity_sigma_begin) / 4
    first_sigma = 0.3
    sigma_interval = 0.1
    velocity_sigma_end = 0.8

    pl.figure(figsize=Utility.figure_size)
    plot_arr = []
    sigma_arr = []
    color_idx = 0
    while first_sigma <= velocity_sigma_end:
        velocity = []
        weight = []
        sigma_arr.append(first_sigma)
        for ele in numpy.linspace(-1, 1, 100):
            velocity.append(ele)
            weight.append(math.exp((-1)*math.pow(ele / first_sigma, 2)))
        l, = pl.plot(velocity, weight, colors[color_idx % len(colors)])
        pl.setp(l, markersize=3)
        plot_arr.append(l)
        first_sigma += sigma_interval
        color_idx += 1
    # pl.title('weight function')
    pl.xlabel('move ability')
    pl.ylabel('weight')
    pl.legend((plot_arr[0], plot_arr[3], plot_arr[1], plot_arr[4], plot_arr[2], plot_arr[5]),
              (r'$\sigma_{MA}=$' + str(sigma_arr[0]), r'$\sigma_{MA}=$' + str(sigma_arr[3]),
               r'$\sigma_{MA}=$' + str(sigma_arr[1]), r'$\sigma_{MA}=$' + str(sigma_arr[4]),
               r'$\sigma_{MA}=$' + str(sigma_arr[2]), r'$\sigma_{MA}=$' + str(sigma_arr[5])),
              loc='upper center', numpoints=1, ncol=3)
    pl.ylim(0, 1.2)
    pl.savefig(u"C:\\Users\Administrator\Desktop\\Figure 1\\weight founction.png", dpi=200)
    pl.show()





def main():
    plot_velocity_weight_mult(0.2, 0.8)

main()

