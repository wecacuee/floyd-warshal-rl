source /etc/profile.d/modules.sh
THISDIR=$(dirname $(readlink -m "${BASH_SOURCE[0]}"))
export MID_DIR=/z/home/dhiman/mid/
source $THISDIR/envsetup/*.sh
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
qsub() {
    $(which qsub) -j oe -o $MID_DIR/$PROJECT_NAME/qsub_\$PBS_JOBID.out $*
}
