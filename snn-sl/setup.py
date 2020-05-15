from setuptools import setup
from setuptools import find_packages
import os

requirement_file = os.path.dirname(
    os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []

if os.path.isfile(requirement_file):
    with open(requirement_file) as f:
        install_requires = f.read().splitlines()

setup(
    name='snn-sl',
    version='0.1.0',
    author='Cleison C. Amorim',
    author_email='cca5@cin.ufpe.br',
    license='GPL-3.0',
    install_requires=install_requires,
    extras_require={
        'tests': ['pytest', 'requests', 'markdown']
    },
    packages=find_packages())
