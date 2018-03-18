"""
* Design requirements:

1. Almost no loss of information.
2. Tree like structure (DAG) with support for shared objects. 
3. Allowing macros for small changes in configurations.
4. Easy copying.

* Why property magic?

The access syntax to the configuration items must be consistent. Since
all types can be wrapped as functions, so functions is one option.
Another option is properties which wrap callables as normal types.

* What is wrong with dictionaries?

Do not support properties

* Proposed Design iteration 1

** ConfClass function to create easy creation class with properties. To
avoid loss of information, the properties can atmost point to other
objects or function call on other properties.

** Shared objects by memoize method utility

* Problems

** Looks ugly.
** Debugging is difficult even for me.
** The lambdas are not given line number.

May be all we need is a different kind of lambdas.

        Conf(_call_func = callable, arg1 = arg1...)()

is same as 

        lambda arg1, ... : callable(arg1)

What you want the configuration to look like

QLearningPlayConf(
_call_func = play,
seed_src = SeedSource,
logger_factory_conf = LoggerFactoryConf,
alg = AlgConf(from_parent = "next_seed logger_factory",
              action_space_conf = ActionSpaceConf( seed_src = SeedSource ),
             ).setter("parent"),
prob = ProbConf,
observer = ObsConf,
logger_factory = LoggerFactoryConf)


"""
import importlib
from argparse import Namespace, ArgumentParser, Action
import inspect
from itertools import chain
from string import Formatter
from collections import namedtuple
import copy
import json

from cog.memoize import MethodMemoizer

def apply_conf(func, conf):
    required_args = list(inspect.signature(func).parameters.keys())
    return func( ** { k : conf[k] for k in required_args } )

class Conf:
    def __init__(self, props=dict(), from_parent = (), parent=None,
                 attrs = dict()):
        self._attributes = attrs
        self._properties = props
        self._from_parent = from_parent
        self._parent = parent

    def _overrides(self, attrs):
        updated_kw = self._attributes.copy()
        updated_kw = dict_update_recursive(updated_kw, attrs)
        return updated_kw

    def copy(self, props=dict(), from_parent = (), parent = None, attrs = dict()):
        updated_props = self._properties.copy()
        updated_props.update(props)
        return type(self)(props = updated_props,
                          from_parent = from_parent or self._from_parent,
                          parent = parent or self._parent,
                          attrs = self._overrides(attrs))

    def setter(self, name):
        return lambda value: self.copy(name = value)

    def keys(self):
        return chain(self._properties.keys(), self._attributes.keys())

    def items(self):
        return chain(self._properties.items(), self._attributes.items())

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    def __getattr__(self, k):
        if k in self._from_parent:
            return getattr(self._parent, k)
        elif k in self._properties:
            try:
                return self._properties[ k ](self)
            except KeyError as e:
                raise AttributeError("self:{} missing k:{}. Error: {}".format(
                    self, k, e))
        else:
            try:
                return self._attributes[ k ]
            except KeyError as e:
                raise AttributeError("self:{} missing k:{}. Error: {}".format(
                    self, k, e))

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

    def argparse_action(self, key, glbls):
        class DefAction(Action):
            def __call__(s, parser, namespace, values, option_string=None):
                setattr(namespace, key, json.loads(values))

        return DefAction

    def parse_remargs(self, remargs, glbls):
        parser = self.parser(self.__class__.__name__)
        for key, defval in vars(self).items():
            if isinstance(defval, Conf):
                defvalcopy = copy.copy(defval)
                parser.add_argument(
                    "--{k}".format(k=key), default=None,
                    action = defvalcopy.argparse_action(key, glbls))
            else:
                #bool_from_str = lambda a : False if a == "False" else bool(a)
                #val_from_default = lambda a: (
                #    bool_from_str(a) if isinstance(v, bool) else type(v)(a))
                #val_from_str = lambda a : glbls.get(a, val_from_default(a))
                parser.add_argument("--{k}".format(k=key), default=None,
                                    type=type(defval))
        args = parser.parse_args(remargs)
        c = self.copy(
            attrs = { k : v for k, v in vars(args).items() if v is not None })
        c.run_checks()
        return c

    def run_checks(self):
        return True

    def __call__(self):
        return apply_conf(self._call_func, self)

    def __repr__(self):
        return "{}".format(vars(self))

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

def NewConfClassInherit(name, props, parents=(Conf,), defaults=dict()):
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
    for v in props.values():
        assert callable(v), "Not implemented"
    dct = { k : property(v) if callable(v) else v for k, v in props.items() }
    if "_defaults" in dct:
        raise NotImplementedError("do not name any prooperty as default")
    dct["_defaults"] = defaults
    return type(name, parents, dct)


def NewConfClass(name, **props):
    return NewConfClassInherit(name, props)


def ConfClass(name, props=dict(), defaults=dict(), **kw):
    return NewConfClassInherit(name, props, defaults=defaults, **kw)


def format_string_from_conf(strformat, obj):
    fieldnames = [fn
                  for lt, fn, fs, conv in Formatter().parse(strformat)
                  if fn is not None]
    formattted_string = strformat.format(**{fn : getattr(obj, fn)
                                            for fn in fieldnames})
    return formattted_string

class ConfTemplate:
    def __init__(self, strformat):
        self.strformat = strformat
    def expand(self, conf):
        return format_string_from_conf(self.strformat, conf)

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
        
