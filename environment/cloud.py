from environment.para_config import *
from .utils import normalize
import random


class Cloud(object):
    def __init__(self):
        self.calc = random.randint(*CloudCalcSpeed)
        self.up_bandwidth = random.randint(
            *CloudUpBandwidth)
        self.down_bandwidth = random.randint(
            *CloudDownBandwidth)

    def obs(self):
        calc = normalize(self.calc, CloudCalcSpeed)
        up_bandwidth = normalize(self.up_bandwidth, CloudUpBandwidth)
        down_bandwidth = normalize(self.down_bandwidth, CloudDownBandwidth)
        return [calc, up_bandwidth, down_bandwidth]
