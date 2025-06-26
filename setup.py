## ............ PROJECT MANAJENT ...........................................##
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(

    name = 'anime-recommendation-system',
    version = '0.1',
    author = 'sarvesh',
    packages = find_packages(),
    install_requires = requirements,
    description = 'A simple anime recommendation system using collaborative filtering',
)