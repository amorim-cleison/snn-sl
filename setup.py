from setuptools import setup
from setuptools import find_packages
from utils.setup_utils import find_requirements

setup(
    name='snn-sl',
    version='0.1.0',
    author='Cleison C. Amorim',
    author_email='cca5@cin.ufpe.br',
    license='GPL-3.0',
    install_requires=find_requirements(__file__),
    extras_require={
        'tests': ['pytest', 'requests', 'markdown']
    },
    packages=find_packages())
