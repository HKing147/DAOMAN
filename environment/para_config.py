GraphSize = (10, 10)  # 地图大小 / 4
Hypotenuse = 2*(GraphSize[0]**2+GraphSize[1]**2)**0.5 # 斜边
# GraphSize = (4000, 4000)  # 地图大小
ΔT = 0.1
Damping = 0.25  # 阻尼
D = 10  # 机器人只能获取到 D 米以内的信息
h = 1e-3  # 信道增益
σ = 1.4142e-6  # 噪声功率
p = 0.2  # 0.2 W
c = 3*10**8  # 光速c = 3*10**8 m/s
w = 0.5  # 移动奖励的权重
EarthToCloud = 100  # 地面到云的距离
DistanceSatelliteToCloud = 1000  # 卫星到云的距离
RobotNum = 3  # 机器人数量
TargetNum = 3  # 目标点数量
ObstacleNum = 5  # 障碍物数量
BSNum = 5  # 基站数量
SatelliteNum = 2  # 卫星数量
# ObstacleR = [10, 100]  # 障碍物半径
ObstacleR = [GraphSize[0]/12, GraphSize[0]/10]  # 障碍物半径
BSR = [GraphSize[0]/6, GraphSize[0]/5]  # 基站覆盖半径
SateR = [GraphSize[0]/3, GraphSize[0]/2]  # 卫星覆盖半径
SateAltitude = [1000*1000, 2000*1000]  # 卫星轨道高度(m)
CollisonEps = 0.5  # 碰撞的界限
ArrivalEps = 0.5  # 到达目标点的界限
VLimit = 10  # 速度限制
ALimit = 10  # 加速度限制
FLimit = 1.0  # 作用力限制


TaskSize = [100, 200]  # 任务大小上下限(MB)
CommSize = [100, 200]  # 通信数据大小上下限(MB)

RobotCalcSpeed = [10, 100]  # 机器人计算速度上下限(MB/s)
RobotUpBandwidth = [10, 100]  # 机器人上行带宽上下限(MBps)
RobotDownBandwidth = [10, 100]  # 机器人下行带宽上下限(MBps)

BSCalcSpeed = [80, 200]  # 基站计算速度上下限(MB/s)
BSUpBandwidth = [100, 1000]  # 基站上行带宽上下限
BSDownBandwidth = [100, 1000]  # 基站下行带宽上下限

SatelliteCalcSpeed = [150, 400]  # 卫星计算速度上下限(MB/s)
SatelliteUpBandwidth = [200, 2000]  # 卫星上行带宽上下限(MBps)
SatelliteDownBandwidth = [200, 2000]  # 卫星下行带宽上下限(MBps)

CloudCalcSpeed = [10000, 20000]  # 云计算速度上下限
CloudUpBandwidth = [500, 10000]  # 云上行带宽上下限
CloudDownBandwidth = [500, 10000]  # 云下行带宽上下限

BandwithBetweenClouds = 1500  # 云之间的通信带宽
