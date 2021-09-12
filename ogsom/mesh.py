'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-11 16:54:20
LastEditTime: 2021-09-12 16:33:31
LastEditors: Qianen
'''
import os
import trimesh
import numpy as np
from .face import Face


class MeshBase(object):
    ''' 基础的网格文件类，用以保存原始数据
    '''

    def __init__(self, trimesh_obj, name):
        '''
        trimesh_obj: 一个trimesh对象
        name: 模型的名字
        '''
        # 一个trimesh对象,用来保存网格数据
        self.trimesh_obj = self.process_mesh(trimesh_obj)
        self.tria = trimesh.ray.ray_triangle.RayMeshIntersector(self.trimesh_obj)
        self.name = name
        if self.trimesh_obj.is_watertight:
            self.center_mass = self.trimesh_obj.center_mass
        else:
            self.center_mass = self.trimesh_obj.centroid

    def ray_test(self, ray_origin, ray_direction, max_distance=None):
        """ 使用rtree进行光线测试,最后的交点按照原点距离排序
            ray_origins: 光线原点, (3,)
            ray_directions: 光线方向, (3,)
        """
        points, _, ids = self.tria.intersects_location(np.reshape(
            ray_origin, (1, 3)), np.reshape(ray_direction, (1, 3)))
        if len(points) == 0:
            return np.array([]), np.array([])
        # 重新排序,使用rtree进行光线检查的结果顺序不是按照原点排序的
        distance = np.linalg.norm(points - np.squeeze(ray_origin), axis=1)
        order = np.argsort(distance)
        if max_distance is not None:
            order = [o for o in order if distance[o] <= max_distance]
        # TODO: 在边界上的点会被计算两次
        return points[order], ids[order], distance[order]

    def intersect_line(self, lineP0, lineP1):
        """ 使用rtree进行相交检测, 直线的两个端点 """
        ray_directions = lineP1 - lineP0
        max_distance = np.linalg.norm(ray_directions)
        return self.ray_test(lineP0, ray_directions, max_distance)

    def process_mesh(self, trimesh_obj):
        ''' 用来预处理mesh数据,这里只是简单的调用了trimesh的预处理程序 '''
        # TODO 可以补上对物体水密性的处理
        trimesh_obj.process()
        return trimesh_obj

    def bounding_box(self):
        max_coords = np.max(self.trimesh_obj.vertices, axis=0)
        min_coords = np.min(self.trimesh_obj.vertices, axis=0)
        return min_coords, max_coords

    def apply_transform(self, matrix):
        tri = self.trimesh_obj.copy()
        tri = tri.apply_transform(matrix)
        return type(self)(tri, self.name, self.mode)

    @staticmethod
    def scale_mesh(mesh, scale):
        vertices = np.array(mesh.vertices) * scale
        mesh_scaled = trimesh.Trimesh(vertices, np.array(mesh.faces))
        return mesh_scaled

    @classmethod
    def from_file(cls, file_path, name=None, scale=None):
        name = name or os.path.splitext(os.path.basename(file_path))[0]
        tri_mesh = trimesh.load_mesh(file_path, validate=True)
        if scale is not None:
            tri_mesh = cls.scale_mesh(tri_mesh, scale)
        return cls(tri_mesh, name=name)

    @classmethod
    def from_data(cls, vertices, triangles, name, normals=None):
        trimesh_obj = trimesh.Trimesh(vertices=vertices,
                                      faces=triangles,
                                      face_normals=normals,
                                      validate=True)
        return cls(trimesh_obj, name=name)


class MeshFace(MeshBase):
    def __init__(self, trimesh_obj, name):
        super().__init__(trimesh_obj, name)
        self.faces = self.get_faces(self.trimesh_obj)

    @staticmethod
    def get_faces(tri_obj):
        v = tri_obj.vertices
        n = tri_obj.face_normals
        faces = []
        for i, fv in enumerate(tri_obj.faces):
            faces.append(Face(np.array(v[fv]), i, n[i]))
        return faces

    def get_face(self, face_id):
        return self.faces[int(face_id)]

    def find_other_contacts(self, c0):
        # 找到其他的接触点
        pass
