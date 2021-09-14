'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 04:43:41
LastEditTime: 2021-09-14 21:15:38
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
        self.valid_sort = self.process_grasps(self.grasps, self.qualities)
        if self.mesh.is_watertight:
            print('check endpoint')
            self.valid_sort = self.check_endpoint(self.mesh, self.grasps, self.valid_sort)

    @classmethod
    def from_obj_file(cls, file_path, step=0.005, scale=0.001):
        mesh = MeshFace.from_file(file_path, scale=scale)
        mesh = mesh.convex_decomposition()
        grasps = cls.sample_grasps(mesh, step)
        qualities = [grasp_quality(g, mesh) for g in grasps]
        return cls(mesh, grasps, qualities)

    @classmethod
    def optimal_v(cls, c0, c1):
        line = c1.point - c0.point
        n0 = c0.normal
        n1 = c1.normal
        nc0 = np.dot(line, n0)
        nc1 = np.dot(-line, n1)
        if nc0 == 0 or nc1 == 0:
            return None
        n0 = -n0 if nc0 > 0 else n0
        n1 = -n1 if nc1 > 0 else n1
        # if np.dot(line, n0) > 0:
        #     n0 = -n0
        # if np.dot(-line, n1) > 0:
        #     n1 = -n1
        # n0 = n0 / np.linalg.norm(n0)
        # n1 = n1 / np.linalg.norm(n1)
        v = n1 - n0
        # print(n0, n1, v, line)
        v = v / np.linalg.norm(v)
        return v

    @classmethod
    def optimize_c1(cls, mesh, c0, c11, max_iter=8):
        c1 = c11
        for j in range(max_iter):
            # print(j)
            vv = cls.optimal_v(c0, c1)
            if vv is None:
                return c11
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
    def process_grasps(grasps, qualities, max_width=0.085, q_th=0.002, c_th=0.01, a_th=np.pi/18):
        """ 剔除无效抓取
        1. 抓取宽度过大的
        2. 质量过小的
        3. 距离过近的
        """
        valid_sort = []
        quality_sort = np.argsort(qualities)[::-1]
        for gi in quality_sort:
            g = grasps[gi]
            if g.contacts_width > max_width:
                continue
            if qualities[gi] < q_th:
                continue
            for gvi in valid_sort:
                gv = grasps[gvi]
                center_dist = np.linalg.norm(g.center - gv.center)
                axis_dist = np.arccos(np.clip(np.abs(g.axis.dot(gv.axis)), 0, 1))
                if center_dist < c_th and axis_dist < a_th:
                    break
            else:
                valid_sort.append(gi)
        return valid_sort

    @staticmethod
    def check_endpoint(mesh, grasps, valid_sort, grasp_width=0.085):
        """ 检查抓取端点是否都在物体外面
        """
        new_valid_sort = []
        for gi in valid_sort:
            g = grasps[gi]
            edp0 = g.center - grasp_width / 2 * g.axis
            edp1 = g.center + grasp_width / 2 * g.axis
            if any(mesh.trimesh_obj.contains([edp0, edp1])):
                continue
            new_valid_sort.append(gi)
        return new_valid_sort
