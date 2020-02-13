#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: setup.py
# @Date: 2019-06-16-21-53
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from setuptools import setup

setup(name='mimo',
      version='0.0.1',
      description='Mixture models inference',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'tikzplotlib', 'joblib'],
      packages=['mimo'],
      zip_safe=False,
      )