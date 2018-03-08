import os
import subprocess
import json
import pickle
import numpy as np
from io import BytesIO
import base64

def ensuredirs(file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file_path

def git_revision(dir_):
    return subprocess.check_output("git rev-parse --short HEAD".split(), cwd=dir_).decode("ascii").strip()

class NumpyEncoder(json.JSONEncoder):
    """
    >>> obj = dict(bye=dict(hi=np.empty((2,))), hello="see")
    >>> obj2 = json.loads(json.dumps(obj, cls=NumpyEncoder),
    ...            object_hook=NumpyEncoder.loads_hook)
    >>> pickle.dumps(obj) == pickle.dumps(obj2)
    True
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            bio = BytesIO()
            np.save(bio, obj)
            return dict(__class__="numpy.ndarray",
                        data=base64.b64encode(bio.getvalue()).decode())
        return super().default(obj)

    @staticmethod
    def loads_hook(dct):
        if dct.get("__class__", "") == "numpy.ndarray":
            return np.load(BytesIO(base64.b64decode(dct["data"])))
        return dct
