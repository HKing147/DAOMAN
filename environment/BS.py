from environment.para_config import *
from environment.circle import Circle
from .utils import normalize
import random


class BS(object):
    def __init__(self, x, y, r):
        self.coverage = Circle(x, y, r)
        self.calc = random.randint(*BSCalcSpeed)
        self.up_bandwidth = random.randint(
            *BSUpBandwidth)
        self.down_bandwidth = random.randint(
            *BSDownBandwidth)

    def obs(self):
        calc = normalize(self.calc, BSCalcSpeed)
        up_bandwidth = normalize(self.up_bandwidth, BSUpBandwidth)
        down_bandwidth = normalize(self.down_bandwidth, BSDownBandwidth)
        return [*self.coverage.obs(), calc, up_bandwidth, down_bandwidth]
