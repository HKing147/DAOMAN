from pprint import pprint
from gym import spaces
from environment.circle import Circle
from environment.robot_cloud import RobotCloud
from environment.robot import Robot
from environment.BS import BS
from environment.satellite import Satellite
from environment.para_config import *
from random import *
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import torch
from torch.distributions import Categorical
from munkres import Munkres, print_matrix, make_cost_matrix

INF = 10  # 惩罚


# 计算两点的距离
def distance(dot1, dot2):
    return ((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2) ** 0.5


# 计算点到线段的距离
def point_to_line_distance(point, line_start, line_end):
    eps = 1e-8

    def sign(x):
        if abs(x) < eps:
            return 0
        return 1 if x > 0 else -1

    def distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def cross_product(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

    def dot_product(a, b):
        return a[0] * b[0] + a[1] * b[1]

    s1 = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    s2 = (point[0] - line_start[0], point[1] - line_start[1])
    s3 = (point[0] - line_end[0], point[1] - line_end[1])

    if line_start[0] == line_end[0] and line_start[1] == line_end[1]:
        return distance(point, line_start)

    if sign(dot_product(s1, s2)) < 0:
        return distance(point, line_start)
    elif sign(dot_product(s1, s3)) > 0:
        return distance(point, line_end)
    else:
        return abs(cross_product(line_start, point, line_end)) / distance(line_start, line_end)


# 计算传输速率
def rate(bandwidth):
    return bandwidth * math.log2(1 + (p * h * h) / (σ * σ))


# 机器人与机器人之间的通信时延
def RobotToRobot(robot1, robot2, dataSize):
    t1 = dataSize / rate(robot1.up_bandwidth)
    t2 = 0
    t3 = dataSize / rate(robot2.down_bandwidth)
    return t1 + t2 + t3


# 机器人与基站之间的通信时延
def RobotToBS(robot, BS, dataSize):
    t1 = dataSize / rate(robot.up_bandwidth)
    t2 = 0
    t3 = dataSize / rate(BS.down_bandwidth)
    return t1 + t2 + t3


# 机器人与卫星之间的通信时延
def RobotToSatellite(robot, satellite, dataSize):
    t1 = dataSize / rate(robot.up_bandwidth)
    t2 = satellite.h / c
    t3 = dataSize / rate(satellite.down_bandwidth)
    return t1 + t2 + t3


# 基站与云之间的通信时延
def BSToCloud(BS, cloud, dataSize):
    t1 = dataSize / rate(BS.up_bandwidth)
    t2 = EarthToCloud / c
    t3 = dataSize / rate(cloud.down_bandwidth)
    return t1 + t2 + t3


# 卫星与云之间的通信时延
def SatelliteToCloud(satellite, cloud, dataSize):
    t1 = dataSize / rate(satellite.up_bandwidth)
    t2 = DistanceSatelliteToCloud / c
    t3 = dataSize / rate(cloud.down_bandwidth)
    return t1 + t2 + t3


# 云之间的通信时延
def CloudToCloud(cloud1, cloud2, dataSize):
    t1 = dataSize / rate(cloud1.up_bandwidth)
    t2 = 0
    t3 = dataSize / rate(cloud2.down_bandwidth)
    return t1 + t2 + t3


class Env():
    def __init__(self, args):
        print("env init...")
        self.obs_normalized = True  # obs归一化
        # set the number of agents
        self.agent_num = args.n_agents  # robot_num
        # set the observation dimension of agents
        self.agent_obs_dim = 57  # 35
        # set the action dimension of agents, here set to a five-dimensional
        self.continuous_action_dim = 2  # [v_x, v_y]
        self.discrete_action_dim = 3  # {0,1,2}
        self.agent_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.agent_obs_dim,),
                dtype=np.float32,
            )]
        self.agent_action_space = [spaces.Box(
            low=-1,
            high=+1,
            shape=(self.continuous_action_dim,),  # v_x, v_y
            dtype=np.float32,
        ), spaces.Discrete(self.discrete_action_dim)]  # offloading_action

        # set the environment dimension of agents
        self.env_obs_dim = self.agent_obs_dim * self.agent_num
        # configure spaces
        self.action_space = [self.agent_action_space] * \
            self.agent_num  # [[],[],...]
        self.observation_space = self.agent_observation_space * self.agent_num
        share_obs_dim = self.agent_obs_dim * self.agent_num
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32,
            )] * self.agent_num

        self.target_num = self.agent_num  # target_num
        self.obstacle_num = 3  # obstacle_num
        self.satellite_num = 2  # satellite_num
        self.BS_num = 3  # BS_num
        self.graph_size = GraphSize
        self.ΔT = ΔT
        self.arrival = [False] * self.agent_num
        self.reset(save_trajectory = True)

    def reset(self, save_trajectory = False, recover = False, mode='train'):
        self.save_trajectory = save_trajectory
        # 回到初始状态
        if recover: # 只有robot会变
            self.robots = copy.deepcopy(self.robots_bak)
            # pprint(f"robots__ = {[robot.coordinate for robot in self.robots]}")
            if save_trajectory:
                self.trajectory = copy.deepcopy(self.trajectory_init_bak)
            # self.targets = copy.deepcopy(self.targets_bak)
        else: # 重新生成
            self.gen_obstacles(mode)
            self.gen_robots(mode)
            self.gen_targets(mode)
            self.gen_BSs(mode)
            self.gen_satellites(mode)
            self.allocate_targets()
        self.offloading_results = []

        return self.obs()

    def close(self):
        return

    # 随机生成障碍物
    def gen_obstacles(self, mode='train'):
        self.obstacles = []
        if mode == 'train':
            for i in range(self.obstacle_num):
                x = uniform(-self.graph_size[0], self.graph_size[0])
                y = uniform(-self.graph_size[1], self.graph_size[1])
                r = uniform(*ObstacleR)
                self.obstacles.append(Circle(x, y, r))
        else:
            x = uniform(-self.graph_size[0], 0)
            y = uniform(-self.graph_size[1], 0)
            r = uniform(*ObstacleR)
            self.obstacles.append(Circle(x, y, r))  # 0
            x = uniform(0, self.graph_size[0])
            y = uniform(-self.graph_size[1], 0)
            r = uniform(*ObstacleR)
            self.obstacles.append(Circle(x, y, r))  # 1
            x = uniform(-self.graph_size[0], 0)
            y = uniform(0, self.graph_size[1])
            r = uniform(*ObstacleR)
            self.obstacles.append(Circle(x, y, r))  # 2

    # 随机生成机器人
    def gen_robots(self, mode='train'):
        self.robots = []
        self.trajectory = []
        if mode == 'train':
            for i in range(self.agent_num):
                while True:
                    x = uniform(-self.graph_size[0], self.graph_size[0])
                    y = uniform(-self.graph_size[1], self.graph_size[1])
                    flag = True
                    for obstacle in self.obstacles:
                        if obstacle.contains([x, y]):
                            flag = False
                    if flag:
                        break
                z, v_x, v_y = 0.0, 0.0, 0.0
                self.robots.append(Robot(x, y, z, v_x, v_y))
                if self.save_trajectory:
                    self.trajectory.append([x, y, z, v_x, v_y])
        else:
            # 将机器人生成在障碍物附近
            for obstacle in self.obstacles:
                obst_x, obst_y, obst_r = obstacle.x, obstacle.y, obstacle.r
                while True:
                    dis = uniform(2 * obst_r, 4 * obst_r)
                    theta = uniform(0, 2 * math.pi)
                    x = obst_x + dis * math.cos(theta)
                    y = obst_y + dis * math.sin(theta)
                    if abs(x) > GraphSize[0] or abs(y) > GraphSize[1]:
                        continue
                    flag = True
                    for obstacle in self.obstacles:
                        if obstacle.contains([x, y]):
                            flag = False
                    if flag:
                        break
                z, v_x, v_y = 0.0, 0.0, 0.0
                self.robots.append(Robot(x, y, z, v_x, v_y))
                if self.save_trajectory:
                    self.trajectory.append([x, y, z, v_x, v_y])
        if self.save_trajectory:
            self.trajectory = np.array(self.trajectory).reshape(
                (-1, self.agent_num, 5))
            self.trajectory_init_bak = copy.deepcopy(self.trajectory)
        self.robots_bak = copy.deepcopy(self.robots)

    # 随机生成目标点
    def gen_targets(self, mode='train'):
        self.targets = []
        arrival_reward = INF  # 到达的奖励
        if mode == 'train':
            for i in range(self.target_num):
                while True:
                    x = uniform(-self.graph_size[0], self.graph_size[0])
                    y = uniform(-self.graph_size[1], self.graph_size[1])
                    flag = True
                    for obstacle in self.obstacles:
                        if obstacle.contains([x, y]):
                            flag = False
                    if flag:
                        break
                self.targets.append([x, y, arrival_reward])
        else:
            for i in range(self.target_num):
                while True:
                    if i == 0:
                        x = uniform(-self.graph_size[0], 0)
                        y = uniform(-self.graph_size[1], 0)
                    elif i == 1:
                        x = uniform(0, self.graph_size[0])
                        y = uniform(-self.graph_size[1], 0)
                    else:
                        x = uniform(-self.graph_size[0], 0)
                        y = uniform(0, self.graph_size[1])
                    flag = True
                    for obstacle in self.obstacles:
                        if obstacle.contains([x, y]):
                            flag = False
                    if flag:
                        break
                self.targets.append([x, y, arrival_reward])

        self.targets_bak = copy.deepcopy(self.targets)

    # 随机生成卫星
    def gen_satellites(self, mode='train'):
        self.Satellites = []
        if mode == 'train':
            for i in range(self.satellite_num):
                x = uniform(-self.graph_size[0], self.graph_size[0])
                y = uniform(-self.graph_size[1], self.graph_size[1])
                r = uniform(*SateR)
                self.Satellites.append(Satellite(x, y, r))
        else:
            for i in range(self.satellite_num):
                if i == 0:
                    x = uniform(-self.graph_size[0]/2, 0)
                    y = uniform(-self.graph_size[1]/2, 0)
                else:
                    x = uniform(0, self.graph_size[0]/4)
                    y = uniform(0, self.graph_size[1]/4)
                r = uniform(*SateR)
                self.Satellites.append(Satellite(x, y, r))

    # 随机生成基站
    def gen_BSs(self, mode='train'):
        self.BSs = []
        if mode == 'train':
            for i in range(self.BS_num):
                x = uniform(-self.graph_size[0], self.graph_size[0])
                y = uniform(-self.graph_size[1], self.graph_size[1])
                r = uniform(*BSR)
                self.BSs.append(BS(x, y, r))
        else:
            for i in range(self.BS_num):
                if i == 0:
                    x = uniform(-self.graph_size[0]/2, self.graph_size[0]/2)
                    y = uniform(-self.graph_size[1]/2, self.graph_size[1]/2)
                elif i == 1:
                    x = uniform(-self.graph_size[0]/2, 0)
                    y = uniform(-self.graph_size[1]/2, 0)
                else:
                    x = uniform(0, self.graph_size[0]/2)
                    y = uniform(0, self.graph_size[1]/2)
                r = uniform(*BSR)
                self.BSs.append(BS(x, y, r))

    def allocate_targets(self, allocated_targets = None):
        if allocated_targets is not None:
            for idx in range(self.target_num):
                self.robots[idx].to = allocated_targets[idx]
        else:
            # matrix = [[5., 9., 1., 4.],
            #         [10., 3., 2., 3.],
            #         [8., 7., 4., 20.]]
            matrix = [[-distance(robot.coordinate, target)  # 加负号
                    for target in self.targets] for robot in self.robots]

            cost_matrix = make_cost_matrix(matrix)
            m = Munkres()
            indexes = m.compute(cost_matrix)
            for row, column in indexes:
                self.robots[row].to = column
        return self.obs()

    def avail_action(self):  # offload
        avail_action_mask = []
        for robot in self.robots:
            mask = []
            for offload_action in range(3):  # 3个卸载动作
                if offload_action == 0:  # 本地
                    mask.append(1)  # 一定可以
                elif offload_action == 1:  # 基站
                    # 判断是否处于基站覆盖范围内
                    f = 0
                    for BS in self.BSs:
                        if BS.coverage.contains(robot.coordinate):
                            f = 1
                            break
                    mask.append(f)
                elif offload_action == 2:  # 卫星
                    # 判断是否处于卫星覆盖范围内
                    f = 0
                    for satellite in self.Satellites:
                        if satellite.coverage.contains(robot.coordinate):
                            f = 1
                            break
                    mask.append(f)
            avail_action_mask.append(mask)
        return avail_action_mask

    # 随机选择action
    def choose_actions_random(self):
        move_actions, offloading_actions = [], []
        for i in range(self.agent_num):
            v_x = uniform(-VLimit, VLimit)
            v_y = uniform(-VLimit, VLimit)
            target = i
            move_actions.append([v_x, v_y, target])
            offloading_actions.append(randint(0, 4))
        return [move_actions, offloading_actions]

    def obs(self):
        # 机器人自身obs  dim: 13
        robots_obs = []
        if self.obs_normalized:
            for robot in self.robots:
                # 与目标和以及其它机器人以及障碍物的相对位置
                relactive_pos = []
                target_id = robot.to
                relactive_pos.append(
                    (robot.coordinate[0]-self.targets[target_id][0]) / GraphSize[0])
                relactive_pos.append(
                    (robot.coordinate[1]-self.targets[target_id][1])/GraphSize[1])

                for other in self.robots:
                    if other == robot:
                        continue
                    relactive_pos.append(
                        (robot.coordinate[0]-other.coordinate[0])/GraphSize[0])
                    relactive_pos.append(
                        (robot.coordinate[1]-other.coordinate[1])/GraphSize[1])
                for obstacle in self.obstacles:
                    relactive_pos.append(
                        (robot.coordinate[0] - obstacle.x)/GraphSize[0])
                    relactive_pos.append(
                        (robot.coordinate[1] - obstacle.y)/GraphSize[1])
                    relactive_pos.append(obstacle.r/Hypotenuse)  # 还要加入障碍物的半径

                # 自身速度、自身位置、目标(相对)位置、其它智能体(相对)位置
                # robots_obs.append(robot.coordinate[:2] + robot.v + relactive_pos)
                # robots_obs.append([x/GraphSize[0] for x in robot.coordinate[:2]] +
                #                   [v/VLimit for v in robot.v] +
                #                   relactive_pos)
                satellites_obs = []
                for satellite in self.Satellites:
                    satellite_obs = satellite.obs()
                    satellite_obs[1] -= robot.obs()[0]  # x
                    satellite_obs[2] -= robot.obs()[1]  # y
                    satellites_obs.extend(satellite_obs)
                BSs_obs = []
                for BS in self.BSs:
                    BS_obs = BS.obs()
                    BS_obs[0] -= robot.obs()[0]  # x
                    BS_obs[1] -= robot.obs()[1]  # y
                    BSs_obs.extend(BS_obs)
                robots_obs.append(robot.obs()+relactive_pos +
                                  satellites_obs+BSs_obs)

        else:
            for robot in self.robots:
                # 与目标和以及其它机器人以及障碍物的相对位置
                relactive_pos = []

                target_id = robot.to
                relactive_pos.append(
                    robot.coordinate[0]-self.targets[target_id][0])
                relactive_pos.append(
                    robot.coordinate[1]-self.targets[target_id][1])

                for other in self.robots:
                    if other == robot:
                        continue
                    relactive_pos.append(
                        robot.coordinate[0]-other.coordinate[0])
                    relactive_pos.append(
                        robot.coordinate[1]-other.coordinate[1])
                for obstacle in self.obstacles:
                    relactive_pos.append(robot.coordinate[0] - obstacle.x)
                    relactive_pos.append(robot.coordinate[1] - obstacle.y)
                    relactive_pos.append(obstacle.r)  # 还要加入障碍物的半径

                # 自身速度、自身位置、目标(相对)位置、其它智能体(相对)位置
                robots_obs.append(
                    robot.coordinate[:2] + robot.v + relactive_pos)

        return robots_obs, self.avail_action()

    def offload_obs(self):
        offload_obs = []
        for robot in self.robots:
            obs = robot.obs()
            for satellite in self.Satellites:
                obs.extend(satellite.obs())
            for BS in self.BSs:
                obs.extend(BS.obs())
            offload_obs.append(obs)
        return offload_obs

    def allocate_obs(self):
        # 机器人自身obs  dim: 13
        robots_obs = []
        for robot in self.robots:
            robots_obs.extend(robot.obs()[:-3])
        for target in self.targets:
            robots_obs.extend(target[:2])
        for obstacle in self.obstacles:
            robots_obs.extend(obstacle.obs())
        for satellite in self.Satellites:
            robots_obs.extend(satellite.obs())
        for BS in self.BSs:
            robots_obs.extend(BS.obs())

        return robots_obs


    # 机器人移动
    def robots_move(self, move_actions):  # action: [f_x,f_y]
        # pprint(f"move_actions = {move_actions}")
        # move_actions += np.random.randn(*move_actions.shape) # 加（环境）噪声
        # pprint(f"move_actions_ = {move_actions}")
        move_actions = np.clip(move_actions, -1, 1)  # 加入噪声之后，再clip在[-1, 1]范围之内
        # pprint(f"move_actions__ = {move_actions}")

        move_actions = move_actions * ALimit
        move_rewards = []
        new_trajectory = []
        out_flag = False
        collision_cnt = [0] * self.agent_num 
        for idx, move_action in enumerate(move_actions):
            # pprint(f"{move_action = }")
            x, y, z = self.robots[idx].coordinate
            v_x, v_y = self.robots[idx].v

            v_x_ = v_x + move_action[0] * self.ΔT
            v_y_ = v_y + move_action[1] * self.ΔT
            v_x_, v_y_ = np.clip([v_x_, v_y_], -VLimit, VLimit)
            x_ = x + 0.5 * (v_x+v_x_) * self.ΔT
            y_ = y + 0.5 * (v_y+v_y_) * self.ΔT
            v_x, v_y = v_x_, v_y_

            r_out = 0.0  # 出地图

            # 限制在地图内
            x_ = min(self.graph_size[0], max(-self.graph_size[0], x_))
            y_ = min(self.graph_size[1], max(-self.graph_size[1], y_))

            # 判断是否碰到障碍物
            r_obstacle = 0
            for i in range(self.obstacle_num):
                if point_to_line_distance((self.obstacles[i].x, self.obstacles[i].y), (x, y), (x_, y_)) <= self.obstacles[i].r:
                    collision_cnt[idx] += 1
                    r_obstacle += -1

            # 判断是否到达目标点
            r_arrival = 0
            targert_id = self.robots[idx].to
            dis = distance(
                [x, y], self.targets[targert_id])
            r_arrival += -dis / GraphSize[0]

            self.robots[idx].coordinate = [x_, y_, z]
            self.robots[idx].v = [v_x, v_y]

            r = r_obstacle + r_arrival + r_out
            move_rewards.append(r)
            if self.save_trajectory:
                new_trajectory.append([x_, y_, z, v_x, v_y])

        # 判断是否撞到其它机器人（并行执行）
        r_collision = -1
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i != j and distance(self.robots[i].coordinate[:2], self.robots[j].coordinate[:2]) < CollisonEps:
                    collision_cnt[i] += 1
                    move_rewards[i] += r_collision

        if self.save_trajectory:
            self.trajectory = np.append(
                self.trajectory, [new_trajectory], axis=0)  # 添加轨迹
        return move_rewards, collision_cnt

    # 机器人移动
    def robots_move_bak(self, move_actions):  # action: [f_x,f_y]
        move_rewards = []
        new_trajectory = []
        out_flag = False
        for idx, move_action in enumerate(move_actions):
            # pprint(f"{move_action = }")
            x, y, z = self.robots[idx].coordinate
            v_x, v_y = self.robots[idx].v

            '''action为速度'''
            x += (v_x + move_action[0]) * self.ΔT / 2.0
            y += (v_y + move_action[1]) * self.ΔT / 2.0
            # 更新速度
            v_x, v_y = move_action[0], move_action[1]

            r_out = 0.0  # 出地图
            if abs(x) > self.graph_size[0] or abs(y) > self.graph_size[1]:
                out_flag = True
                r_out = -Hypotenuse
            # 限制在地图内
            x = min(self.graph_size[0], max(-self.graph_size[0], x))
            y = min(self.graph_size[1], max(-self.graph_size[1], y))

            # 判断是否碰到障碍物
            r_obstacle = 0
            for i in range(self.obstacle_num):
                if self.obstacles[i].contains((x, y)):
                    r_obstacle += -Hypotenuse

            r_arrival = []
            for i in range(self.target_num):
                distances = []
                for j in range(self.agent_num):  # 计算目标点i与各个机器人的距离
                    dis = distance(
                        self.targets[i][:2], self.robots[j].coordinate)
                    if dis <= ArrivalEps:  # 到达了该目标点, 加上奖励
                        distances.append(-dis)
                    else:
                        distances.append(-dis)
                r_arrival.append(max(distances))

            self.robots[idx].coordinate = [x, y, z]
            self.robots[idx].v = [v_x, v_y]

            r = []
            for i in range(self.target_num):
                r.append((r_arrival[i] + r_obstacle + r_out))
            move_rewards.append(r)
            if self.save_trajectory:
                new_trajectory.append([x, y, z, v_x, v_y])

        # 判断是否撞到其它机器人（并行执行）
        r_collision = -Hypotenuse
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i != j and distance(self.robots[i].coordinate[:2], self.robots[j].coordinate[:2]) < CollisonEps:
                    for k in range(self.target_num):
                        move_rewards[i][k] += r_collision

        if self.save_trajectory:
            self.trajectory = np.append(
                self.trajectory, [new_trajectory], axis=0)  # 添加轨迹
        return move_rewards

    # 机器人任务卸载
    def task_offloading(self, offloading_actions):
        '''
            0: 本地
            1: 基站
            2: 卫星
        '''
        penalty = 2 * max(TaskSize[1]/RobotCalcSpeed[0], # local
                        CommSize[1]/RobotUpBandwidth[0] + CommSize[1] / BSDownBandwidth[0] + TaskSize[1]/BSCalcSpeed[0], # BS
                        CommSize[1]/RobotUpBandwidth[0] + SateAltitude[1]/c + CommSize[1]/SatelliteDownBandwidth[0] + TaskSize[1]/SatelliteCalcSpeed[0]) # SA
        offloading_rewards = []
        offloading_result = []
        average_cost = [] # 处理单位数据花费的时间
        for idx, offloading_action in enumerate(offloading_actions):
            robot = self.robots[idx]
            t = 0.0
            t_map_local = 0.0  # 机器人间直接交换地图的时延
            for i in range(self.agent_num):
                if i != idx and distance(robot.coordinate, self.robots[i].coordinate) < D:
                    t_map_local += RobotToRobot(self.robots[i],
                                                robot, robot.task.commSize)
            f = False
            if offloading_action == 0:  # 本地（通信+计算）
                t_comp = robot.task.taskSize/robot.calc
                t = t_map_local+t_comp
                f = True
            elif offloading_action == 1:  # 基站
                # 先判断是否处于基站覆盖范围内
                f = False
                for BS in self.BSs:
                    if BS.coverage.contains(robot.coordinate):
                        t_comm_task = RobotToBS(robot, BS, robot.task.taskSize)
                        t_comp = robot.task.taskSize/BS.calc
                        t = t_map_local+t_comm_task+t_comp
                        f = True
                        break
                # 不处于基站覆盖范围
                if f == False:
                    t = penalty
            elif offloading_action == 2:  # 卫星
                # 先判断是否处于卫星覆盖范围内
                f = False
                for satellite in self.Satellites:
                    if satellite.coverage.contains(robot.coordinate):
                        t_comm_task = RobotToSatellite(
                            robot, satellite, robot.task.taskSize)
                        t_comp = robot.task.taskSize/satellite.calc
                        t = t_map_local+t_comm_task+t_comp
                        f = True
                        break
                # 不处于卫星覆盖范围
                if f == False:
                    t = penalty
            average_cost.append(t / robot.task.taskSize)
            t /= penalty  # => [0, 1]
            # print(f"{idx = } {offloading_action = } {f = } {t = }")
            offloading_result.append([offloading_action, f])
            offloading_rewards.append(-t)
            self.robots[idx].generate_task()  # 重新生成任务
        self.offloading_results.append(offloading_result)

        return offloading_rewards, average_cost

    def step(self, actions):
        move_actions, offload_actions = actions
        offload_rewards, average_cost = self.task_offloading(offload_actions)
        move_rewards, collision_cnt = self.robots_move(move_actions)
        rewards = [[w*move_rewards[i], (1-w)*offload_rewards[i]]
                   for i in range(self.agent_num)]
        terminal = sum(self.arrival) == self.agent_num
        terminals = [terminal] * self.agent_num
        return *self.obs(), rewards, terminals, {"collision_cnt":collision_cnt,"average_cost": average_cost}

    def render(self, ax=None):
        self.save_video(save_file='video.gif')
        return

    def save_video(self, save_file=None, interval=100, repeat=True):
        if self.save_trajectory == False:
            return
        # save initial state
        import pickle
        robots, targets, obstacles, BSs, Satellites = self.robots_bak, self.targets, self.obstacles, self.BSs, self.Satellites
        with open(save_file + '.txt', 'wb') as f:
            f.write(pickle.dumps([robots, targets, obstacles, BSs, Satellites]))
        '''
        data: 数据
        save_file: 保存的文件名
        interval: 每帧之间的时间间隔
        repeat: 循环播放
        '''
        from pprint import pprint
        pprint(f"robots = {[robot.coordinate for robot in self.robots]}")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{xcolor}'

        trajectory = self.trajectory
        colors = ['green', 'red', 'blue', 'orange']
        markers = ['o', '*', 'p', 'h']
        fig, ax = plt.subplots(figsize=(10, 10))

        def init():  # 初始plot
            ax.cla()  # 如果循环播放的话，会清空之前的
            ax.set_xlim(-self.graph_size[0], self.graph_size[0])
            ax.set_ylim(-self.graph_size[1], self.graph_size[1])
            ax.set_xticks([])  # 取消刻度
            ax.set_yticks([])

            title = ""
            self.offloading_results.append(
                [[0, True], [0, True], [0, True]])  # 最后一个时刻
            for i in range(len(trajectory[0])):
                # # 画卫星
                for Satellite in self.Satellites:
                    circle = plt.Circle((Satellite.coverage.x, Satellite.coverage.y),
                                        Satellite.coverage.r, fill=True, color='#55efc4', alpha=0.1)
                    ax.add_artist(circle)
                # # 画地面基站
                for BS in self.BSs:
                    circle = plt.Circle((BS.coverage.x, BS.coverage.y),
                                        BS.coverage.r, fill=True, color='#16a085', alpha=0.8)
                    ax.add_artist(circle)
                # 画障碍物
                for obstacle in self.obstacles:
                    circle = plt.Circle((obstacle.x, obstacle.y),
                                        obstacle.r, fill=True, color='gray')
                    ax.add_artist(circle)
                # 画起点
                ax.plot(trajectory[0, i, 0], trajectory[0, i, 1],
                        color=colors[i], marker="^", markersize=12, label=f"Agent_{i}")
                target_id = self.robots[i].to
                # 画目标点
                ax.plot(self.targets[target_id][0], self.targets[target_id][1], marker="*",
                        color=colors[i], markersize=15)

        def animate(i):  # 更新函数
            # print(i)
            if i == 0:
                init()  # 循环播放会回到初始状态
            else:
                title = ""
                for j in range(len(trajectory[0])):
                    ax.plot(trajectory[i - 1:i + 1, j, 0], trajectory[i - 1:i + 1, j, 1],  # i-1: 前一个点， i: 当前点
                            color=colors[j], marker=markers[j], alpha=0.3, label=f"Agent_{j}")


        ani = animation.FuncAnimation(fig, animate, range(
            len(trajectory)), init_func=init, interval=interval, repeat=repeat)
        # plt.show()
        if save_file is not None:
            ani.save(save_file)
            print(f"save {save_file} successfully!")

        plt.close()
