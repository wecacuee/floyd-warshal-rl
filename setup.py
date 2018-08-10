import os
import sys
import pip
import subprocess
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
from argparse import Namespace

import setuptools
import sys
import logging
LOG = logging.getLogger(__name__)

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


@lru_cache()
def CONF():
    return Namespace(project_name="floyd_warshall_rl",
                     mid_dir="/z/home/dhiman/mid/",
                     project_mid_dir="{mid_dir}/{project_name}",
                     mode    = "devel", # not implemented
                     builddir="build",
                     install_requires=[
                         "numpy>=1.14.0",
                         "opencv>=3.4.0",
                         "matplotlib>=2.1.2",
                         "ipython>=6.2.1",
                         "pytorch>=0.4.1",
                         "torchvision",
                         "gym[mujoco]",
                         "gym[atari]",
                         "atari-py>=0.1.1",
                         "PyOpenGL",
                     ])

def relpath(fpath, rootdir=Path(__file__).parent):
    return  rootdir / Path(fpath)

@contextmanager
def changed_environ(**kwargs):
    oldenv = os.environ
    os.environ.update(kwargs)
    yield
    os.environ = oldenv

def pip_install(conf, packages):
    with changed_environ(PYTHONUSERBASE=str(builddir(conf))):
        for pkg in packages:
            pip.main(["install", "--user", pkg])

def py_version_dir():
    return "python{0.major}.{0.minor}".format(sys.version_info)

def pip_provided(conf, packages):
    with relpath(Path("envsetup") / "pipload.sh").open("w") as f:
        f.write("""export PYTHOHPATH={pypath}""".format(
            pypath=":".join(
                [str(builddir(conf) / "lib" /
                    py_version_dir() / "site-packages")])))


def builddir(conf):
    return relpath(conf.builddir, rootdir=project_mid_dir(conf))

def module_supported():
    return subprocess.call(["bash", "-c", "module avail"], stderr=open("/dev/null", "w"))

def module_map():
    return {
        "numpy>=1.14.0" : "numpy/py3.6/1.14.0",
        "opencv>=3.4.0" : "opencv/3.4.0",
        "matplotlib>=2.1.2" : "matplotlib/py3.6/2.1.2",
        "ipython>=6.2.1" : "ipython/py3.6/6.2.1",
        "pytorch>=0.4.1" : "pytorch/py3.6/0.4.1",
        "torchvision" : "pytorch/py3.6/0.4.1"
        }

def module_package_available(pkg):
    try:
        module_name = module_map()[pkg]
    except KeyError:
        LOG.info("module: {pkg} name not known".format(pkg=pkg))
        return False
    ret = len(subprocess.check_output(
        ["bash", "-c", "module avail {pkg}".format(pkg=module_name)],
        stderr=subprocess.STDOUT
    ).strip())
    if not ret:
        LOG.info("module:{pkg} not available".format(pkg=pkg))
    else:
        LOG.info("module:{pkg} available".format(pkg=pkg))
    return ret

def module_provided(packages):
    with relpath(Path("envsetup") / "moduleload.sh").open("w") as f:
        f.write(
            """module load miniconda3/4.5.1 cuda/8.0.61 cudnn/8.0-v6.0 """
            + " ".join([module_map()[p] for p in packages
                        if module_package_available(p)]))
        f.write("\n")

def touch(filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print("touching {}".format(str(filepath)))
    with filepath.open("w") as f:
        f.write("")


def project_mid_dir(conf):
    return Path(conf.project_mid_dir.format(mid_dir=conf.mid_dir,
                                            project_name=conf.project_name))

def setup_install_requires(install_requires):
    return [p for p in install_requires
            if not module_package_available(p)]

def setup():
    conf = CONF()
    if module_supported():
        module_provided(conf.install_requires)

    setuptools.setup(
        name=conf.project_name,
        description='Floyd warshall RL',
        author='Vikas Dhiman',
        url='git@opticnerve.eecs.umich.edu:dhiman/floyd-warshall-rl.git',
        author_email='dhiman@umich.edu',
        version='0.1.0',
        license='MIT',
        classifiers=(
            'Development Status :: 3 - Alpha',
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ),
        packages=setuptools.find_packages(),
        install_requires=setup_install_requires(conf.install_requires),
        dependency_links=[
            "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl"
        ],
        python_requires='~=3.6',
        entry_points={
            'console_scripts': [
                'floyd_warshall_rl=fwrl.conf.default:main',
                'fwrl-4-room-gw=fwrl.conf.four_room_grid_world:main',
                'fwrl-qlnet-cartpole=fwrl.conf.qlnet_cartpole:demo',
                'fw-simple-gw-play=fwrl.conf.fw_simple_gw:main',
            ],
        },
        include_package_data = True,
        package_data={
            # If any package contains *.txt or *.rst files, include them:
            '': ['*.txt'],
        },
    )

setup()
