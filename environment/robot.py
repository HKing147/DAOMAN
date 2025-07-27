from environment.para_config import *
from environment.task import Task
from environment.cloud import Cloud
from .utils import normalize
import random


class Robot(object):
    def __init__(self, x, y, z, v_x, v_y):
        self.coordinate = [x, y, z]
        self.v = [v_x, v_y]
        self.mass = 1.0  # 质量
        self.calc = random.randint(*RobotCalcSpeed)
        self.up_bandwidth = random.randint(
            *RobotUpBandwidth)
        self.down_bandwidth = random.randint(
            *RobotDownBandwidth)
        # self.cloud = Cloud()
        self.task = None
        self.generate_task()
        self.to = -1  # 要到的目标点

    # 生成任务
    def generate_task(self):
        self.task = Task()

    # 完成协作任务
    def complete_task(self, task):
        return task.taskSize / self.calc

    def obs(self):
        observation = []
        coordinate = [self.coordinate[0]/GraphSize[0],
                      self.coordinate[1]/GraphSize[1]]
        v = [self.v[0]/VLimit, self.v[1]/VLimit]
        calc = normalize(self.calc, RobotCalcSpeed)
        up_bandwidth = normalize(self.up_bandwidth, RobotUpBandwidth)
        down_bandwidth = normalize(self.down_bandwidth, RobotDownBandwidth)
        observation.extend(coordinate)
        observation.extend(v)
        observation.append(calc)
        observation.append(up_bandwidth)
        observation.append(down_bandwidth)
        # observation.extend(self.cloud.obs())
        observation.extend(self.task.obs())
        observation.append(self.to)
        return observation
