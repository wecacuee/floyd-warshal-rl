"""
This modules describes a minilanguage that is meant to describe what
instead of how.
If this was scheme, it would have been trivial. But I am trying to
stay as pythonic as possible using introspection, lambdas and
properties.
"""
import importlib
from argparse import Namespace, ArgumentParser
import inspect
from string import Formatter
from collections import namedtuple
import copy

from cog.memoize import MethodMemoizer

"""
A pair of function and configuration object that can be applied using apply_conf
"""
LazyApplyable = namedtuple("LazyApplyable", "func conf".split())

def apply_conf(func, conf):
    required_args = list(inspect.signature(func).parameters.keys())
    return func( ** { k : conf[k] for k in required_args } )

MEMOIZE_METHOD = MethodMemoizer()
class Conf(Namespace):
    def __init__(self, **kw):
        defaults = self.defaults()
        updated_kw = dict_update_recursive(defaults, kw)
        MEMOIZE_METHOD.init_obj(self)
        super().__init__(**updated_kw)

    def defaults(self):
        return dict()

    def items(self):
        return vars(self).items()

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    @classmethod
    def parser(cls, default_config):
        parser = ArgumentParser()
        parser.add_argument("--config", default=default_config)
        return parser

    @classmethod
    def import_class(cls, name):
        module_name, class_ = name.strip().split(":")
        module = importlib.import_module(module_name)
        return getattr(module, class_)

    @classmethod
    def parse_all_args(cls, default_config, args, glbls={}, **kwargs):
        c, remargs = cls.parser(default_config).parse_known_args(args)
        conf = cls.import_class(c.config)(**kwargs)
        return conf.parse_remargs(remargs, glbls=glbls)

    def parse_remargs(self, remargs, glbls):
        parser = self.parser(self.__class__.__name__)
        for k, v in vars(self).items():
            if isinstance(v, Conf):
                conf_from_args = lambda a : v.parse_remargs(a, glbls)
                parser.add_argument(
                    "--{k}".format(k=k), default=None, type=conf_from_args)
            else:
                bool_from_str = lambda a : False if a == "False" else bool(a)
                val_from_default = lambda a: (
                    bool_from_str(a) if isinstance(v, bool) else type(v)(a))
                val_from_str = lambda a : glbls.get(a, val_from_default(a))
                parser.add_argument("--{k}".format(k=k), default=None,
                                    type=val_from_str)
        args = parser.parse_args(remargs)
        self.__init__(
            **{ k : v for k, v in vars(args).items() if v is not None })
        self.run_checks()
        return self

    def run_checks(self):
        return True

    def apply_func(self):
        return apply_conf(self.func, self)

def dict_update_recursive(ddest, dsrc):
    for k, v in dsrc.items():
        if isinstance(v, (dict, Conf)):
            if k in ddest: 
                ddest[k] = dict_update_recursive(ddest[k], v)
            else:
                ddest[k] = copy.copy(v)
        else:
            ddest[k] = v

    return ddest

def NewConfClass(name, **kwargs):
    """
    Defines a new class with name = name and
    initialize kwargs attributes as property objects.
    All arguments to the kwargs should be callables (usually lambda functions).
    This is useful for redirecting configurations to same values
    ```
    MyConf = NewConfClass("MyConf",
                abc = lambda s : s.bleh.abc,
                xyz = lambda s : s._parent.blah.xyz)
    ```

    is equivalent to

    ```
    class MyConf(Conf):
        @property
        def abc(s):
            return s.bleh.abc

        @property
        def xyz(s):
            return s._parent.blah.xyz
    ```
    """
    for v in kwargs.values():
        assert callable(v), "Not implemented"
    return type(name, (Conf,),
                { k : property(v) if callable(v) else v
                  for k, v in kwargs.items() })


def format_string_from_obj(strformat, obj):
    fieldnames = [fn
                  for lt, fn, fs, conv in Formatter().parse(strformat)
                  if fn is not None]
    formattted_string = strformat.format(**{fn : getattr(obj, fn)
                                            for fn in fieldnames})
    return formattted_string


def serialize_list(l):
    return list(map(serialize_any, l))

def serialize_dict(d):
    return { k : serialize_any(v) for k, v in d.items() }


def serialize_any(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return serialize_list(obj)
    elif isinstance(obj, dict):
        return serialize_dict(obj)
    elif isinstance(obj, Conf):
        return serialize_dict(vars(obj))
    else:
        #print(f"serialize_any : ignoring {type(obj)}")
        return {}

def MultiConfGen(name, confs):
    conf_dict = (confs
                 if isinstance(confs, dict)
                 else { str(k) : v for k, v in enumerate(confs) })
    conf_keys = conf_dict.keys()
    return type(name, (Conf, ),
                dict(
                    defaults = lambda self: dict(
                        func = lambda confs: [confs[k].apply_func() for k in conf_keys],
                        confs = Conf(**conf_dict))))
        
