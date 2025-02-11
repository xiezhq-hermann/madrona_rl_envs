#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='madronarlenvs',
      version='0.0.1',
      description='MadronaRLEnvs',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'torch',
          'numpy',
          'tqdm',
          'stable-baselines3',
          'hanabi_learning_environment',
      ],
      )
