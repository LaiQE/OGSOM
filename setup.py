'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-13 07:55:23
LastEditTime: 2021-09-14 18:48:09
LastEditors: Qianen
'''
from setuptools import setup, find_packages

requirements = [
    'numpy',
    'trimesh'
]

setup(name='ogsom',
      version='1.0',
      description='',
      author='LaiQE',
      author_email='laiqe@zju.edu.cn',
      url='https://github.com/LaiQE/OGSOM',
      packages=find_packages(),
      install_requires=requirements
      )
