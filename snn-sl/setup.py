from setuptools import setup
from setuptools import find_packages

setup(
    name='snn-sl',
    version='0.1.0',
    author='Cleison C. Amorim',
    author_email='cca5@cin.ufpe.br',
    license='GPL-3.0',
    install_requires=['numpy>=1.9.1', 
                      'scipy>=0.14', 
                      'keras>=2.3.1', 
                      'tensorflow>=2.1.0',
                      'matplotlib>=3.1.3'],
    extras_require={
        'tests': ['pytest', 'requests', 'markdown'],
    },
    packages=find_packages())
