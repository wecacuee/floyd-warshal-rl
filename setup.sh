. /etc/profile.d/modules.sh
THISDIR=$(dirname $(readlink -m "${BASH_SOURCE[0]}"))
export MID_DIR=/z/home/dhiman/mid/
. $THISDIR/envsetup/*.sh
MCMD=/z/sw/Modules/$MODULE_VERSION/bin/modulecmd
eval `$MCMD bash use /z/home/dhiman/wrk/common/modulefiles/`
eval `$MCMD bash load miniconda3/4.5.1 cuda/8.0.61 cudnn/8.0-v6.0 numpy/py3.6/1.14.0 opencv/3.4.0 matplotlib/py3.6/2.1.2 ipython/py3.6/6.2.1 pytorch/py3.6/0.4.1 pytorch/py3.6/0.4.1`
echo "MODULEPATH is:"
echo $MODULEPATH
echo "--------"
export PROJECT_NAME=floyd_warshall_rl
PIPDIR=$MID_DIR/$PROJECT_NAME/build/
PYPATH=$PIPDIR/lib/python3.6/site-packages/
if [[ "$PYTHONPATH" != *"$PYPATH"* ]]; then
    export PYTHONPATH=$PYPATH:$PYTHONPATH
fi
if [[ "$PATH" != *"$PIPDIR/bin"* ]]; then
    export PATH=$PIPDIR/bin:$PATH
fi
PYTHONUSERBASE=$PIPDIR pip install --user --upgrade -e .
MJPATH=$HOME/.mujoco/mjpro150/bin
if [[ "$LD_LIBRARY_PATH" != *"$MJPATH"* ]]; then
    export LD_LIBRARY_PATH=$MJPATH:$LD_LIBRARY_PATH
fi
