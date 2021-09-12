'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-12 14:24:49
LastEditTime: 2021-09-12 15:13:09
LastEditors: Qianen
'''

import numpy as np


class Point(object):
    def __init__(self, coordinate) -> None:
        super().__init__()
        self.coordinate = np.array(coordinate)

    @property
    def x(self):
        return self.coordinate[0]

    @property
    def y(self):
        return self.coordinate[1]

    @property
    def z(self):
        return self.coordinate[2]

    @property
    def coor(self):
        return self.coordinate


class FacePoint(Point):
    def __init__(self, coordinate, face_id) -> None:
        super().__init__(coordinate)
        self.face_id = face_id
