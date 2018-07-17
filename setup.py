from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='floyd-warshall-rl',
      install_requires=[
      ],
      description='Floyd warshall RL',
      author='Vikas Dhiman',
      url='git@opticnerve.eecs.umich.edu:dhiman/floyd-warshall-rl.git',
      author_email='dhiman@umich.edu',
      version='0.0.0')
