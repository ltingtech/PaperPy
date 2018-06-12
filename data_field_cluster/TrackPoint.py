#  _*_coding:utf-8_*_


class TrackPoint:
    def __init__(self, lon, lat, time):
        self.lon = lon
        self.lat = lat
        self.time = time
        self.time_str = ''
        self.velocity = 0
        self.potential = 0
        self.distance = 0
        self.is_interpolated = False
        self.is_core_point = False
        self.is_border_point = False
        self.is_noise_point = False
