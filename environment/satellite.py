from environment.para_config import *
from environment.circle import Circle
from .utils import normalize
import random


class Satellite(object):
    def __init__(self, x, y, r):
        self.h = random.randint(*SateAltitude)  # 卫星轨道高度(m)
        self.coverage = Circle(x, y, r)
        self.calc = random.randint(
            *SatelliteCalcSpeed)
        self.up_bandwidth = random.randint(
            *SatelliteUpBandwidth)
        self.down_bandwidth = random.randint(
            *SatelliteDownBandwidth)

    def obs(self):
        h = normalize(self.h, SateAltitude)
        calc = normalize(self.calc, SatelliteCalcSpeed)
        up_bandwidth = normalize(self.up_bandwidth, SatelliteUpBandwidth)
        down_bandwidth = normalize(self.down_bandwidth, SatelliteDownBandwidth)
        return [h, *self.coverage.obs(),  calc, up_bandwidth, down_bandwidth]
