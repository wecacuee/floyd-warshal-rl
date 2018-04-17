"""
* Design requirements:

1. Almost no loss of information before execution.
2. Tree like structure (DAG) with support for shared objects. 
3. Opacity of functions and other objects.

There three main features
    * Keywords that depend on other keywords.
    * Inverting argument
    * __getattr__ for easy interface access for func defaults and partials


"""
import importlib
from argparse import Namespace, ArgumentParser, Action
import inspect
from itertools import chain
from string import Formatter
from collections import namedtuple, OrderedDict
import copy
import json
import functools
from functools import partial
from contextlib import contextmanager

from cog.memoize import method_memoizer

def empty_to_none(v):
    return None if v is inspect._empty else v

def func_need_args(func):
    params = inspect.signature(func).parameters
    return { k : empty_to_none(p.default) for k, p in params.items()
             if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD }

def apply_conf(func, conf, args=()):
    args_from_conf = { k : getattr(conf, k) for k in args }
    return func(**args_from_conf)

def func_kwonlydefaults(func):
    return {k : p.default for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect._empty}

class KWProp:
    def __init__(self, func):
        if not callable(func):
            raise ValueError("fget should be callable")
        self.fget = func
        functools.update_wrapper(self, func)

    def __call__(self, *a, **kw):
        return self.fget(*a, **kw)

    def __getattr__(self, a):
        return getattr(self.fget, a)

    def __repr__(self):
        return "{self.__class__.__name__}({self.fget})".format(self=self)

#KWProp = property

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


class FuncProp:
    """
    Makes the function evaluation opaque. The function definition
    decides how to return the value rather then user.

    >>> f = FuncProp(n = 1, np1 = KWProp(lambda s : s.n + 1))
    >>> f.np1
    2
    """
    post_process = process_kwprop
    def __init__(self, **kw):
        self.attrs = kw

    def get(self, attr):
        return self.post_process(self.attrs[attr])

    def __getattr__(self, attr):
        try:
            return self.get(attr)
        except KeyError as e:
            raise AttributeError(e)

def update_signature(wrapper, wrapped, args, kwargs):
    """
    Updates the signature of wrapper with wrapped if 

    >>> 
    """
    oldsig = inspect.signature(wrapped)
    oldp = oldsig.parameters
    params = OrderedDict( (k, p.replace(default = arg))
                          for arg, (k, p) in zip(args, oldp.items()))
    kwparam = OrderedDict( (k , oldp[k].replace(default = kwargs[k]))
                            for k, v in kwargs.items() )
    inspect.signature(wrapper).replace(
        parameters = OrderedDict(chain(params.items(), kwparam.items())))
    return wrapper

def wrap_funcs(func, wrapper):
    return (wrapper(func)
            if callable(func)
            else wrapper(func.fget)
            if isinstance(func, KWProp)
            else func)

def undo_wrap_funcs(func, wrapper):
    return  (KWProp(wrapper)
             if isinstance(func, KWProp)
             else wrapper)

def identity(func):
    return func

class FuncAttr:
    """
    Lets you access the default kwargs of function as it's attributes.

    >>> def two(one = 1): return one + 1
    >>> FuncAttr(two).one
    1
    """
    def __init__(self, func):
        self.func = func
        self._partial = dict()
        functools.update_wrapper(self, func)

    def _wrap_child(self, v):
        return type(self)(func = v) if callable(v) else v

    def _wrap_kwdefaults(self, kwd):
        return { k : self._wrap_child(v) for k, v in kwd.items() }

    @property
    def partial(self):
        self._partial = (
            self._partial or self._wrap_kwdefaults(func_kwonlydefaults(self.func)))
        return self._partial

    def __call__(self, *args, **kw):
        return self.func(*args, **dict(self.partial, **kw))

    def __getattr__(self, k):
        return getattr(self.func, k, self.partial[k])

    def __setattr__(self, k, v):
        if (k in """func _func_defaults_ _wr_kw _partial _post_process
                    _undo_post_process""".split() or
            k.startswith("__") and k.endswith("__")):
            object.__setattr__(self, k, v)
        else:
            self._partial[k] = v

    def __repr__(self):
        return """{self.__class__.__name__}({self.func.__name__},
        **{self.partial})""".format(self=self)


def extended_kwprop(func):
    """
    >>> p = KWProp
    >>> two = lambda one = 1: one + 1
    >>> two()
    2
    >>> ptwo = partial(two, one = p(lambda s : s.zero + 1))
    >>> two2 = extended_kwprop(ptwo)
    >>> two2(zero = 10)
    12
    """
    attrs = func_kwonlydefaults(func)
    @functools.wraps(func)
    def wrapper(*args, **kw):
        kwprops_m = FuncProp(**dict(attrs, **kw))
        return apply_conf(partial(func, *args), kwprops_m, attrs.keys())

    return wrapper

def xargs_(func, expect_args=(), *args, **kwargs):
    pfunc = partial(func, *args, **kwargs)
    @functools.wraps(pfunc)
    def wrapper(conf, *a, **kw):
        return apply_conf(partial(pfunc, *a, **kw), conf, expect_args)

    return wrapper

def xargs(func, expect_args=(), *args, **kwargs):
    return KWProp(xargs_(func, expect_args, *args, **kwargs))

def xargmem(func, expect_args=(), *args, **kwargs):
    return KWProp(method_memoizer(xargs_(func, expect_args, *args, **kwargs)))

    
