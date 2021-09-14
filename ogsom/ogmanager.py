'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 04:43:41
LastEditTime: 2021-09-14 14:15:16
LastEditors: Qianen
'''
import numpy as np
from .contact import Contact
from .grasp import Grasp3D
from .mesh import MeshFace
from .quality import grasp_quality


class OGManager(object):
    def __init__(self, mesh, grasps, g_qualities) -> None:
        super().__init__()
        self.mesh = mesh
        self.grasps = grasps
        self.qualities = np.array(g_qualities)
        self.quality_sort = np.argsort(g_qualities)[::-1]

    @classmethod
    def from_obj_file(cls, file_path, step=0.005, scale=0.001):
        mesh = MeshFace.from_file(file_path, scale=scale)
        mesh = mesh.convex_decomposition()
        grasps = cls.sample_grasps(mesh, step)
        qualities = [cls.grasps_quality(mesh, g) for g in grasps]
        return cls(mesh, grasps, qualities)

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
    def optimize_c1(cls, mesh, c0, c11, max_iter=8):
        c1 = c11
        for j in range(max_iter):
            # print(j)
            vv = cls.optimal_v(c0, c1)
            cc0 = Contact(c0._point, c0.normal, vv, mesh.center_mass)
            cc1s = mesh.find_other_contacts(cc0)
            if len(cc1s) == 0:
                return c11
            # 取和原来的c1最近的那个点作为新的c1
            cc1_d = [np.linalg.norm(c1.point-c.point) for c in cc1s]
            cc1_i = np.argmin(cc1_d)
            cc1 = cc1s[cc1_i]
            if abs(abs(np.dot(cc1.normal, c1.normal)) - 1) < 1e-3:
                return cc1
            c1 = cc1
        print('优化失败')
        return c11

    @classmethod
    def sample_grasps(cls, mesh, step):
        grasps = []
        for face in mesh.faces:
            face_points = face.sample(step)
            print('--------', len(face_points), face.id)
            for p in face_points:
                c0 = Contact.from_facepoint(mesh, p, mesh.center_mass)
                c1s = mesh.find_other_contacts(c0)
                for c1 in c1s:
                    n_c1 = cls.optimize_c1(mesh, c0, c1)
                    v = n_c1.point - c0.point
                    v = v / np.linalg.norm(v)
                    n_c0 = Contact(c0._point, c0.normal, v, mesh.center_mass)
                    n_c1._grasp_direction = -v
                    grasps.append(Grasp3D(n_c0, n_c1))
        return grasps

    @staticmethod
    def grasps_quality(mesh, grasp):
        try:
            quality = grasp_quality(grasp, mesh)
        except Exception as e:
            print('quality calculate fault, ', e)
            quality = 0
            raise e
        return quality
