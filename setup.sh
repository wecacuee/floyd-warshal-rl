source /etc/profile.d/modules.sh
THISDIR=$(dirname $(readlink -m "${BASH_SOURCE[0]}"))
export MID_DIR=/z/home/dhiman/mid/
source $THISDIR/envsetup/*.sh
PYPATH=$THISDIR/py
if [[ "$PYTHONPATH" != *"$PYPATH"* ]]; then
    export PYTHONPATH=$PYPATH:$PYTHONPATH
fi
PYTHONUSERBASE=$MID_DIR/floyd_warshall_rl/build/ pip install --user --upgrade .
