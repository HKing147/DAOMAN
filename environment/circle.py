from environment.para_config import *


class Circle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def contains(self, pos):
        return self.r**2 >= (pos[0]-self.x)**2+(pos[1]-self.y)**2

    def obs(self):
        return [self.x / GraphSize[0], self.y/GraphSize[1], self.r/Hypotenuse]
