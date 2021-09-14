'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 07:49:42
LastEditTime: 2021-09-13 08:52:20
LastEditors: Qianen
'''
import numpy as np
import trimesh
from .color import cnames


def mesh_add_color(mesh, color='red'):
    vision = mesh.copy()
    if isinstance(color, (str)):
        color = trimesh.visual.color.hex_to_rgba(cnames[color])
    vision.visual.face_colors = color
    vision.visual.vertex_colors = color
    return vision


def line_to_mesh(p0, p1, matrix=np.eye(4), radius=0.0025, color='red', width_offset=0):
    def vector_to_rotation(vector):
        z = np.array(vector)
        z = z / np.linalg.norm(z)
        x = np.array([1, 0, 0])
        x = x - z*(x.dot(z)/z.dot(z))
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.c_[x, y, z]
    width = np.linalg.norm(p0-p1)
    axis = (p0 - p1) / width
    vision = trimesh.creation.capsule(width + width_offset, radius)
    rotation = vector_to_rotation(axis)
    trasform = np.eye(4)
    trasform[:3, :3] = rotation
    trasform[:3, 3] = p1 - axis * width_offset * 0.5
    vision.apply_transform(trasform)
    if isinstance(color, (str)):
        color = trimesh.visual.color.hex_to_rgba(cnames[color])
    vision.visual.face_colors = color
    vision.visual.vertex_colors = color
    vision = vision.apply_transform(matrix)
    return vision


def point_to_mesh(point, matrix=np.eye(4), radius=0.003, color='black'):
    point_vision = trimesh.creation.uv_sphere(radius)
    trasform = np.eye(4)
    trasform[:3, 3] = point
    point_vision.apply_transform(trasform)
    if isinstance(color, (str)):
        color = trimesh.visual.color.hex_to_rgba(cnames[color])
    point_vision.visual.face_colors = color
    point_vision.visual.vertex_colors = color
    point_vision = point_vision.apply_transform(matrix)
    return point_vision


def grasp_to_mesh(grasp, matrix=np.eye(4), radius=0.0025, color='red'):
    return line_to_mesh(grasp.endpoints[0], grasp.endpoints[1], matrix, radius, color)


def grasp_center_to_mesh(grasp, matrix=np.eye(4), radius=0.003, color='black'):
    return point_to_mesh(grasp.center, matrix, radius, color)
