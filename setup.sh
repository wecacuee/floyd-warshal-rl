THISDIR=$(dirname $(readlink -m "${BASH_SOURCE[0]}"))
export MID_DIR=/z/home/dhiman/mid/
source envsetup/*.sh
PYPATH=$THISDIR/py
if [ $PYTHONPATH != *"$PYPATH"* ]; then
    export PYTHONPATH=$PYPATH:$PYTHONPATH
fi
