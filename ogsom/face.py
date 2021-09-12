'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-12 14:21:56
LastEditTime: 2021-09-12 15:13:24
LastEditors: Qianen
'''
import numpy as np
from .point import FacePoint


class Face(object):

    def __init__(self, points, face_id, normal=None) -> None:
        super().__init__()
        self.id = face_id
        self.points = [FacePoint(p.coordinate, self.id) for p in self.points]
        if normal is not None:
            self.normal = normal
        else:
            self.normal = np.cross(self.edge0, self.edge1)
            self.normal = self.normal / np.linalg.norm(self.normal)

    @property
    def edge0(self):
        return self.points[1].coor - self.points[0].coor

    @property
    def edge1(self):
        return self.points[2].coor - self.points[0].coor

    @property
    def points_matrix(self):
        return np.array([p.coordinate for p in self.points])

    @property
    def center(self):
        return np.mean(self.points_matrix, axis=0)

    def sample(self, step, include_center=True):
        edge0 = self.edge0
        edge1 = self.edge1
        edge0_len = np.linalg.norm(edge0)
        edge1_len = np.linalg.norm(edge1)
        k = -edge1_len / edge0_len
        b = edge1_len
        vx = edge0 / edge0_len
        vy = edge1 / edge1_len
        # 使用两个边的方向向量作为基底
        # 可以看成是在一个长方形的一半里面采样
        sample_points = []
        if include_center:
            sample_points.append(self.center)
        for x in np.arange(0, edge0_len, step)[1:]:
            y_max = k * x + b
            for y in np.arange(0, y_max, step)[1:]:
                coor = vx * x + vy * y
                sample_points.append(FacePoint(coor, self.id))
        return sample_points
