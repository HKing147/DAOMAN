import random
from .utils import normalize
from environment.para_config import *


class Task(object):
    def __init__(self):
        self.taskSize = random.randint(*TaskSize)  # 任务数据大小
        self.commSize = random.randint(*CommSize)  # 通信数据大小

    def obs(self):
        taskSize = normalize(self.taskSize, TaskSize)
        commSize = normalize(self.commSize, CommSize)
        return taskSize, commSize
