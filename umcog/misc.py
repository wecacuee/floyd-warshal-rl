import os
import subprocess
import json
import pickle
import operator
from io import BytesIO
import base64
import builtins
from itertools import tee, islice, repeat
from functools import partial, update_wrapper, reduce

import numpy as np

def ensuredirs(file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file_path

def git_revision(dir_):
    return subprocess.check_output("git rev-parse --short HEAD".split(), cwd=dir_).decode("ascii").strip()

class NumpyEncoder(json.JSONEncoder):
    """
    >>> obj = dict(bye=dict(hi=np.empty((2,))), hello="see", abe = 0)
    >>> obj3 = obj.copy()
    >>> obj3["abe"] = np.uint8(0)
    >>> obj2 = json.loads(json.dumps(obj3, cls=NumpyEncoder),
    ...            object_hook=NumpyEncoder().loads_hook)
    >>> pickle.dumps(obj) == pickle.dumps(obj2)
    True
    """
    def isinstance(self, obj):
        return isinstance(obj, np.ndarray)

    def dumps(self, obj):
        bio = BytesIO()
        np.save(bio, obj)
        return dict(__class__="numpy.ndarray",
                    data=base64.b64encode(bio.getvalue()).decode())

    def isinstance_scalar(self, obj):
        return isinstance(obj, (np.bool_, np.int8, np.int16, np.int32,
                                np.int64, np.uint8, np.uint16,
                                np.uint32, np.uint64, np.float16,
                                np.float32, np.float64, np.complex128, np.complex64))

    def encode_scalar(self, obj):
        return np.asscalar(obj)
        
    def default(self, obj):
        if self.isinstance(obj):
            return self.dumps(obj)
        elif self.isinstance_scalar(obj):
            return self.encode_scalar(obj)
        return super().default(obj)

    def is_dct_instance(self, dct):
        return dct.get("__class__", "") == "numpy.ndarray"

    def loads(self, dct):
        return np.load(BytesIO(base64.b64decode(dct["data"])))

    def loads_hook(self, dct):
        if self.is_dct_instance(dct):
            return self.loads(dct)
        return dct

class DictEncoder(json.JSONEncoder):
    """
    >>> obj = [ 0, 1, 3, "hi", { (0, 1) :  3,  0 : 2, (0, 2, 3) : 2 }]
    >>> obj2 = json.loads(DictEncoder().encode(obj),
    ...                   object_hook = DictEncoder().loads_hook)
    >>> pickle.dumps(obj) == pickle.dumps(obj2)
    True
    """
    def isinstance(self, obj):
        return (isinstance(obj, dict) and
            not all( isinstance(k, str) for k in obj.keys() ))

    def preprocess(self, obj):
        if self.isinstance(obj):
            obj = dict( __class__ = "pydict",
                        data = [ (k, self.preprocess(v)) for k, v in obj.items() ])
        elif isinstance(obj, (list, tuple)):
            obj = [self.preprocess(e) for e in obj]
        elif isinstance(obj, dict):
            obj = { k : self.preprocess(v) for k, v in obj.items() }

            
        return obj

    def encode(self, obj):
        obj = self.preprocess(obj)
        return super().encode(obj)

    def is_dct_instance(self, dct):
        return dct.get("__class__", "") == "pydict"

    def loads(self, dct):
        return { (tuple(k) if isinstance(k, list) else k) : v
                 for k, v in dct["data"] }

    def loads_hook(self, dct):
        if self.is_dct_instance(dct):
            return self.loads(dct)
        return dct


class ChainedEncoders(json.JSONEncoder):
    def __init__(self, encoders=None, **kw):
        self.encoders = encoders
        super().__init__(**kw)
                

    def default(self, obj):
        for enc in self.encoders:
            if enc.isinstance(obj):
                return enc.dumps(obj)
        super().default(obj)

    def loads_hook(self, dct):
        for enc in self.encoders:
            if enc.is_dct_instance(dct):
                return enc.loads(dct)
        return dct

## Functools
def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a, b = tee(iterable)
    return zip(islice(a, None, None, 2), islice(b, 1, None, 2))

class kwcompose:
    def __init__(self,
                 *fs,
                 apply_one=lambda ret, f: f(**ret)):
        if len(fs) < 2: raise ValueError("Need at least one function to compose")
        self.fs = fs
        self.apply_one = apply_one
        update_wrapper(self, fs[-1])

    def __call__(self, **kw):
        return reduce(self.apply_one, self.fs[:-1], self.fs[-1](**kw))


compose = partial(kwcompose, apply_one = lambda ret, f: f(ret))


def dictzip(kwiterables):
    keys, values = zip(*kwiterables.items())
    return (dict(zip(keys, v)) for v in zip(*values))

def kwmap(function, **kwiterables):
    return (function(**kw) for kw in dictzip(kwiterables))

def prod(seq):
    return reduce(operator.mul, seq, 1)

