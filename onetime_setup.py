#!/usr/bin/python3
import os
import sys
import pip
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
from argparse import Namespace

@lru_cache()
def CONF():
    return Namespace(project_name="floyd_warshall_rl",
                     mid_dir="/z/home/dhiman/mid/",
                     project_mid_dir="{mid_dir}/{project_name}",
                     mode    = "devel", # not implemented
                     builddir="build",
                     pypkgs=""""numpy opencv matplotlib""".split(),
                     pkg_install_func=module_provided, # pip_install
                     setupdone="{builddir}/project_setup_done")

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
    return Path("/etc/profile.d/modules.sh").exists()

def module_provided(conf, packages):
    with relpath(Path("envsetup") / "moduleload.sh").open("w") as f:
        f.write("/etc/profile.d/modules.sh")
        f.write("\n")
        f.write("""module load miniconda3/4.5.1 numpy/py3.6 cuda/8.0.61 cudnn/8.0-v6.0 opencv/3.4.0 matplotlib/py3.6/2.1.2 ipython/py3.6/6.2.1  pytorch/py3.6/0.2.0""")
        f.write("\n")

def touch(filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print("touching {}".format(str(filepath)))
    with filepath.open("w") as f:
        f.write("")

def setupdone(conf):
    return Path(conf.setupdone.format(builddir=builddir(conf)))

def project_mid_dir(conf):
    return Path(conf.project_mid_dir.format(mid_dir=conf.mid_dir,
                                            project_name=conf.project_name))

def setup_project(conf):
    conf.pkg_install_func(conf, conf.pypkgs)

if __name__ == '__main__':
    conf = CONF()
    if not setupdone(conf).exists():
        setup_project(conf)
        touch(setupdone(conf))
    else:
        print("project already setup at {}".format(setupdone(conf)))

