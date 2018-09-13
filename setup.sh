export PROJECT_NAME=floyd-warshall-rl

THISFILE="${BASH_SOURCE[0]}"
if [[ -z "$THISFILE" ]]; then
    THISDIR=$(pwd)
else
    THISDIR=$(dirname $(readlink -m $THISFILE))
fi

if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module use /z/home/dhiman/wrk/common/modulefiles/
    source $THISDIR/envsetup/moduleload.sh
fi

export MID_DIR=/z/home/dhiman/mid/
PIPDIR=$MID_DIR/$PROJECT_NAME/build/
PYPATH=$PIPDIR/lib/python3.6/site-packages/
if [[ "$PYTHONPATH" != *"$PYPATH"* ]]; then
    export PYTHONPATH=$PYPATH:$PYTHONPATH
fi
if [[ "$PATH" != *"$PIPDIR/bin"* ]]; then
    export PATH=$PIPDIR/bin:$PATH
fi

PYTHONUSERBASE=$PIPDIR pip install --user --upgrade --process-dependency-links -e $THISDIR
python setup.py test
MJPATH=$HOME/.mujoco/mjpro150/bin
if [[ "$LD_LIBRARY_PATH" != *"$MJPATH"* ]]; then
    export LD_LIBRARY_PATH=$MJPATH:$LD_LIBRARY_PATH
fi
