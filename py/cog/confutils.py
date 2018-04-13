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
import functools
from contextlib import contextmanager

from cog.memoize import method_memoizer

def empty_to_none(v):
    return None if v is inspect._empty else v

def func_need_args(func):
    params = inspect.signature(func).parameters
    required_args = list(params.keys())
    return required_args

def apply_conf(func, conf):
    required_args = func_need_args(func)
    args_from_conf = { k : conf.get(k) for k in required_args
                       if k in conf }
    return func(**args_from_conf)

def func_kwonlydefaults(func):
    return {k : p.default for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect._empty}

class KWProp:
    __slots__ = ["fget"]
    def __init__(self, fget):
        if not callable(fget):
            raise ValueError("fget should be callable")
        self.fget = fget
    def __repr__(self):
        return "{self.__class__.__name__}({self.fget})".format(self=self)

def props2attrs(props, propclass=KWProp):
    return { k : propclass(v) for k, v in props.items() }

def prop_exec_handler(self, p):
    return p.fget(self)

def prop_noexec_handler(s, p):
    return p.fget if isinstance(p.fget, type(s)) else type(s)(p.fget)

def process_kwprop(self, val, prop_handler=prop_exec_handler):
    if isinstance(val, KWProp):
        return prop_handler(self, val)
    else:
        return val

process_kwprop_noexec = functools.partial(
    process_kwprop,
    prop_handler=prop_noexec_handler)


@contextmanager
def confmode(wrapfunc):
    change_post_proc(wrapfunc, process_kwprop_noexec)
    yield wrapfunc
    change_post_proc(wrapfunc, process_kwprop)

class WrapFunc:
    def __init__(self, func,
                 post_process = process_kwprop,
                 **kwargs):
        self.func = func
        self.post_process = post_process
        self.attrs = kwargs
        self._func_defaults = None
        functools.update_wrapper(self, func)

    def change_post_proc(self, post_process):
        self.post_process = post_process
        for k, v in chain(self.func_defaults.items(), self.attrs.items()):
            if isinstance(v, KWProp) and isinstance(v.fget, WrapFunc):
                v.fget.change_post_proc(post_process) 


    def partial(self, **kwargs):
        self.attrs.update(kwargs)
        return self

    def copy(self, func = None, **kw):
        c = type(self)
        return c(func or self.func, **dict(self.attrs, **kw))

    def __call__(self, **kw):
        c = self.copy(**kw)
        return apply_conf(c.func, c)

    @property
    def func_defaults(self):
        self._func_defaults = self._func_defaults or func_kwonlydefaults(self.func)
        return self._func_defaults

    def _getattr(self, attr):
        if attr in self.attrs:
            return self.attrs[attr]
        elif attr in self.func_defaults:
            return self.func_defaults[attr]
        else:
            raise AttributeError(attr)

    def __contains__(self, attr):
        return attr in self.attrs or attr in self.func_defaults

    def get(self, attr):
        return self.post_process(self, self._getattr(attr))

    def __getattr__(self, attr):
        return self.get(attr)

    def __repr__(self):
        return " ".join("""{self.__class__.__name__}(
            {self.func.__name__},
            func_defaults = {self.func_defaults},
            **{self.attrs}
            )""".split()).format(self=self)


def from_conf(f):
    return WrapFunc(lambda conf, f: apply_conf(func, conf), f = f)

WrapFuncNoExec = functools.partial(WrapFunc, post_process =
                                   process_kwprop_noexec)


class WFuncFB(WrapFunc):
    def __init__(self, func, **kwargs):
        self.fb_attrs = []
        self.fb = None
        super().__init__(func, **kwargs)

    def copy(self, func = None, **kwargs):
        c = WrapFunc.copy(self, func = func, **kwargs)
        c.fb_attrs = self.fb_attrs.copy()
        c.fb = self.fb
        return c
        
    def expects(self, attrs):
        self.fb_attrs = attrs
        return self

    def __call__(self, fb = None, **kwargs):
        self.fb = fb or self.fb
        return WrapFunc.__call__(self, **kwargs)

    def __contains__(self, attr):
        return (attr in self.fb_attrs and attr in self.fb) or  WrapFunc.__contains__(
            self, attr)

    def get(self, attr):
        if attr in self.fb_attrs:
            return self.fb.get(attr)
        else:
            return WrapFunc.get(self, attr)
            
WFuncFBNoExec = functools.partial(WFuncFB,
                                  post_process = process_kwprop_noexec)

def KWFuncExp(func, exp_attrs, **kwargs):
    return KWProp(WFuncFB(func, **kwargs).expects(exp_attrs))

def args_to_recursive_conf(argparseobj, confclass=WFuncFB, sep="."):
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
            if isinstance(defval, WFuncFB):
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

def dict_update_recursive(ddest, dsrc, recursetypes=(dict, WFuncFB)):
    for k, v in dsrc.items():
        if isinstance(v, recursetypes):
            if k in ddest: 
                dict_update_recursive(ddest[k], v)
            else:
                ddest[k] = copy.copy(v)
        else:
            ddest[k] = v

    return ddest

def NewConfClassInherit(name, props, parents=(WFuncFB,), defaults=dict()):
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
    elif isinstance(obj, WFuncFB):
        return serialize_dict(vars(obj))
    else:
        #print(f"serialize_any : ignoring {type(obj)}")
        return {}

def MultiConfGen(name, confs):
    conf_dict = (confs
                 if isinstance(confs, dict)
                 else { str(k) : v for k, v in enumerate(confs) })
    conf_keys = conf_dict.keys()
    return type(name, (WFuncFB, ),
                dict(
                    defaults = lambda self: dict(
                        func = lambda confs: [confs[k].apply_func() for k in conf_keys],
                        confs = WFuncFB(**conf_dict))))


class ConfOneArg(WFuncFB):
    def __init__(self, argname="arg1", **kwargs):
        self.argname = argname
        WFuncFB.__init__(self, lambda : 0, **kwargs)

    def __call__(self, arg1):
        self.attrs[self.argname] = arg1
        return WFuncFB.__call__(self)

MEMOIZE_METHOD = ConfOneArg(argname="method", retfunc=method_memoizer,
                            **func_kwonlydefaults(method_memoizer))


    
