'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 04:43:41
LastEditTime: 2021-09-13 06:02:48
LastEditors: Qianen
'''
import numpy as np
from .contact import Contact


class OGManager(object):
    def __init__(self, mesh, grasps=None) -> None:
        super().__init__()
        self.mesh = mesh
        self.grasps = grasps

    @classmethod
    def optimal_v(cls, c0, c1):
        line = c1.point - c0.point
        n0 = c0.normal
        if np.dot(line, n0) > 0:
            n0 = -n0
        n1 = c1.normal
        if np.dot(-line, n1) > 0:
            n1 = -n1
        # n0 = n0 / np.linalg.norm(n0)
        # n1 = n1 / np.linalg.norm(n1)
        v = n1 - n0
        v = v / np.linalg.norm(v)
        return v

    @classmethod
    def optimize_c1(cls, mesh, c0, c11, max_iter=3):
        c1 = c11
        for _ in range(max_iter):
            vv = cls.optimal_v(c0, c1)
            cc0 = Contact(c0.point, c0.normal, vv)
            cc1s = mesh.find_other_contacts(cc0)
            if len(cc1s) == 0:
                return c11
            # 取和原来的c1最近的那个点作为新的c1
            cc1_d = [np.linalg.norm(c1.point-c.point) for c in cc1s]
            cc1_i = np.argmin(cc1_d)
            cc1 = cc1s[cc1_i]
            if abs(np.dot(cc1.normal, c1.normal) - 1) < 1e-3:
                return cc1
            c1 = cc1
        return c11

    @classmethod
    def sample_grasp(cls, mesh, step):
        for face in mesh.faces:
            face_points = face.sample(step)
            for p in face_points:
                c0 = Contact.from_facepoint(mesh, p)
                c1s = mesh.find_other_contacts(c0)
                for c1 in c1s:
                    n_c1 = cls.optimize_c1(mesh, c0, c1)
