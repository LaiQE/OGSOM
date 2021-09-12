'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 06:40:04
LastEditTime: 2021-09-13 06:56:25
LastEditors: Qianen
'''
import numpy as np


class Grasp3D(object):
    """ 点接触二指夹爪抓取点模型类
    一般来说3D抓取定义在物体坐标系下
    """

    def __init__(self, c0, c1):
        self.c0 = c0
        self.c1 = c1
        self.center = (c0.point + c1.point) / 2
        self.axis = c0.grasp_direction

    @property
    def contacts(self):
        return [self.c0, self.c1]
