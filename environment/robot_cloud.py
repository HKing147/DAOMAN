from environment.robot import Robot
from environment.cloud import Cloud
from environment.task import Task
from environment.para_config import *


class RobotCloud(object):
    def __init__(self):
        self.robot = Robot()
        self.cloud = Cloud()
        self.task = Task()
        self.channel = 0  # 信道 0: 未占用 1: 已占用

    # 生成任务
    def generate_task(self):
        self.task = Task()

    # 本地发送数据的时间
    def transmit_local(self):
        return self.task.commSize / self.robot.transmit_power

    # 本地接收数据的时间
    def recevie_local(self, dataSize):
        return dataSize/self.robot.received_power

    # 本地处理任务
    def process_task_local(self):
        return self.robot.complete_task(self.task)

    # 将任务卸载到云端的传输时间
    def upload_task_to_cloud(self):
        taskSize = self.task.taskSize
        return taskSize/self.robot.transmit_power+taskSize/BandwithBetweenRobotAndCloud+self.recevie_cloud(taskSize)

    # 云端发送数据的时间
    def transmit_cloud(self):
        return self.task.commSize / self.cloud.transmit_power

    # 云端接收数据的时间
    def recevie_cloud(self, dataSize):
        return dataSize/self.cloud.received_power

    # 云端处理任务
    def process_task_cloud(self):
        return self.task.taskSize/self.cloud.calc
