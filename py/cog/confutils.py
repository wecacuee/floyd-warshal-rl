"""
* Design requirements:

1. Almost no loss of information. When a function is executed it eats
  its arguments and may result in output from which the input cannot
  be recovered. This is not to mean that functions should not be
  executed, but all the arguments to the function should be kept around.
2. Tree like structure (DAG) with support for shared objects. 
3. Allowing macros for small changes in configurations. As long we
   keep the configuration as a DAG, we can manipulate the DAG to change configuration.
4. Easy copying.

* A function is best written, when most of it is written in the
  kwargs. Because that makes it overrideable.

* Another technique of programming can be properties like kwargs,
  where some of the kwargs are just lambdas that depend on the
  locals() inside the function.

* Why property magic?

The access syntax to the configuration items must be consistent. Since
all types can be wrapped as functions, so functions is one option.
Another option is properties which wrap callables as normal types.

* What is wrong with dictionaries?

Do not support properties.

* Proposed Design

** ConfClass function to create easy creation class with properties. To
avoid loss of information, the properties can atmost point to other
objects or function call on other properties.

** Shared objects by delegating the shared object to appropriate level of parent hood.

* Features
** Unification for function and non function properties
** 

"""
import importlib
from argparse import Namespace, ArgumentParser, Action
import inspect
from itertools import chain
from string import Formatter
from collections import namedtuple
import copy
import json

from cog.memoize import method_memoizer

def apply_conf(func, conf):
    required_args = list(inspect.signature(func).parameters.keys())
    return func( ** { k : getattr(conf, k) for k in required_args } )

def func_kwonlydefaults(func):
    return {k : p.default for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect._empty}

def props2attrs(props):
    return { k : property(v) for k, v in props.items() }

def call_if_prop(val, args):
    return val.fget(*args) if isinstance(val, property) else val


class KWFunc:
    def __init__(self,
                 attrs = dict(),
                 retkey = "retval",
                 exec_enabled = True,
                 **kwargs):
        self.retkey = retkey
        self.exec_enabled = exec_enabled
        self.attrs = dict()
        self.partial(attrs, **kwargs)

    def partial(self, attrs=dict(), **kwargs):
        self.attrs = dict(self.attrs, **dict(attrs, **kwargs))
        return self

    def copy(self, **kwargs):
        c = type(self)()
        return self.merge(c, **kwargs)

    def merge(self, c, attrs=dict(), **kwargs):
        KWFunc.__init__(c, attrs = dict(self.attrs, **dict(attrs)), **kwargs)
        return c

    def __call__(self, **kwargs):
        self.partial(**kwargs)
        return getattr(self, self.retkey)

    def enable_exec(self, val=True):
        self.exec_enabled = val
        for k, v in self.attrs.items():
            if isinstance(v, property) and isinstance(v.fget, KWFunc):
                v.fget.enable_exec(val)
            if isinstance(v, KWFunc):
                v.enable_exec(val)

    def handle_prop(self, prop):
        func = prop.fget
        if self.exec_enabled:
            return func(self)
        else:
            return func
            

    def __getattr__(self, k):
        try:
            val = self.attrs[k]
        except KeyError as e:
            raise AttributeError("missing k:{}. Error: {}; \n self:{} "
                                    .format(k, e, self.attrs))
        return self.handle_prop(val) if isinstance(val, property) else val

    def __repr__(self):
        return self.__class__.__name__ + "(" + "\n".join(
            (k + ": " + repr(getattr(self, k))) for k in "retkey exec_enabled attrs".split()
        ) + ")"
    

def KWFuncArgs(retval, **kwargs):
    return KWFunc(props = dict(retval = lambda s: apply_conf(retval, s)),
                  **kwargs)

class Conf(KWFunc):
    def __init__(self,
                 props = dict(),
                 attrs = dict(),
                 retfunckey = "retfunc",
                 default_retval = lambda s: apply_conf(getattr(s, s.retfunckey), s),
                 fallback = None,
                 **kwargs):
        self.retfunckey = retfunckey
        self.fallback = fallback
        attrs = dict(attrs, **props2attrs(props))
        super().__init__(attrs = attrs, **kwargs)
        self.attrs.setdefault(self.retkey, property(default_retval))

    def merge(self, c, **kwargs):
        attrs = kwargs.pop("attrs", dict())
        props = kwargs.pop("props", dict())
        attrs = dict(props2attrs(props), **attrs)
        KWFunc.merge(self, c, attrs = attrs, **kwargs)
        c.retfunckey = kwargs.pop("retfunckey", self.retfunckey)
        c.fallback = kwargs.pop("fallback", self.fallback)
        return c

    def setfb(self, fallback):
        assert fallback is not self
        self.fallback = fallback
        return self

    def __call__(self, fallback=None, **kwargs):
        if fallback:
            self.fallback = fallback
        return KWFunc.__call__(self, **kwargs)

    def __getattr__(self, k):
        if k in "__contains__ __getitem__ __setitem__ keys items".split():
            return getattr(self.attrs, k)
        else:
            try:
                return super().__getattr__(k)
            except AttributeError as e:
                if self.fallback and not (k.startswith("__") and k.endswith("__")):
                    try: 
                        return getattr(self.fallback, k)
                    except AttributeError as e2:
                        raise e
                raise e

def makeconf(func, confclass=Conf, attrs=dict(), **kw):
    kwargs = dict(dict(__name__ = func.__name__), **kw)
    return confclass(attrs = dict(func_kwonlydefaults(func), **attrs),
                     retfunc = func,
                     **kwargs)


def args_to_recursive_conf(argparseobj, confclass=Conf, sep="."):
    for key, value in vars(argparseobj).items():
        keys = key.split(sep)


def conf_to_key_type_default(
        conf, 
        typeconvs = {
            type(True) : lambda str_: False if str_ == "False" else bool(str_)},
        typeconv_from_default = type,
        glbls=dict()):
    key_conv_default = dict()
    items = conf.items()
    for k, val in items:
        if isinstance(val, type(conf)):
            key_conv_default.update(
                { ".".join((k, k2)) : (".".join((k, k2)), c2, d2)
                  for _, (k2, c2, d2) in 
                  conf_to_key_type_default(val,
                                           typeconvs, typeconv_from_default).items()})
        elif isinstance(val, (float, int, str, bool)):
            vtype = type(val)
            conv = typeconv_from_default(val)
            key_conv_default[k] = (k, typeconvs.get(vtype, conv), val)
        elif id(val) in map(id, glbls.values()):
            key_conv_default[k] = (k, lambda s: glbls[s], val)
        else:
            # Memoized keys are being skipped
            #print("Skipping key {} of type {} {}".format(k, v, vconf))
            pass
            
    return key_conv_default

def add_arguments_from_conf(parser, key_conv_default):
    for key, (_, conv, default) in key_conv_default.items():
        parser.add_argument("--" + key, default=default, type=conv)


class ConfFromDotArgs:
    def __init__(self, conf):
        self.conf = conf

    def default_parser(self, parser = None):
        parser = parser or ArgumentParser()
        if self.conf.fallback:
            key_conv_default = conf_to_key_type_default(self.conf.fallback)
        else:
            key_conv_default = dict()
        key_conv_default.update(conf_to_key_type_default(self.conf))
        add_arguments_from_conf(parser, key_conv_default)
        return parser

    def parse_from_args(self, sys_argv, parser = None):
        return self.default_parser(parser).parse_known_args(sys_argv)

    def conf_from_args(self, args):
        return self.conf.copy(attrs=vars(args))

    def from_args(self, sys_argv, parser = None):
        args, remargv = self.parse_from_args(sys_argv, parser)
        return self.conf_from_args(args)


class ConfFromArgs:
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

    def __repr__(self):
        return "{}".format(vars(self))

def dict_update_recursive(ddest, dsrc, recursetypes=(dict, Conf)):
    for k, v in dsrc.items():
        if isinstance(v, recursetypes):
            if k in ddest: 
                dict_update_recursive(ddest[k], v)
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
                xyz = lambda s : s.parent.blah.xyz)
    ```

    is equivalent to

    ```
    class MyConf(Conf):
        @property
        def abc(s):
            return s.bleh.abc

        @property
        def xyz(s):
            return s.parent.blah.xyz
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


class ConfOneArg(Conf):
    def __init__(self, argname="arg1", **kwargs):
        self.argname = argname
        Conf.__init__(self, **kwargs)

    def __call__(self, arg1):
        self.attrs[self.argname] = arg1
        return Conf.__call__(self)

MEMOIZE_METHOD = ConfOneArg(argname="method", retfunc=method_memoizer,
                            **func_kwonlydefaults(method_memoizer))


    
