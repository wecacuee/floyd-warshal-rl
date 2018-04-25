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
from argparse import ArgumentParser
import inspect
from itertools import chain
import json
import functools

from cog.memoize import method_memoizer

def empty_to_none(v):
    return None if v is inspect._empty else v

def func_need_args(func):
    params = inspect.signature(func).parameters
    return { k : empty_to_none(p.default) for k, p in params.items()
             if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD }

def nondefault_argnames(func):
    params = inspect.signature(func).parameters
    return [k for k, p in params.items() if
            (p.default is inspect.Parameter.empty
             and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD))]

def apply_conf(func, conf, args=()):
    args_from_conf = { k : getattr(conf, k) for k in args }
    return func(**args_from_conf)

def func_kwonlydefaults_from_sig(func):
    return {k : p.default
            for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect._empty}

def func_kwonlydefaults(func):
    if isinstance(func, functools.partial):
        kw = func_kwonlydefaults(func.func)
        kw.update(func.keywords)
    else:
        kw = func_kwonlydefaults_from_sig(func)
    return kw


class KWProp:
    _kwprop = object()
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
    @classmethod
    def isinstance(cls, obj):
        return hasattr(obj, '_kwprop') and obj._kwprop is cls._kwprop

#KWProp = property

def props2attrs(props, propclass=KWProp):
    return { k : propclass(v) for k, v in props.items() }

def prop_exec_handler(self, p):
    return p.fget(self)

def prop_noexec_handler(s, p):
    return p.fget if isinstance(p.fget, type(s)) else type(s)(p.fget)

def process_kwprop(self, val, prop_handler=prop_exec_handler):
    if KWProp.isinstance(val):
        return prop_handler(self, val)
    else:
        return val


process_kwprop_noexec = functools.partial(
    process_kwprop,
    prop_handler=prop_noexec_handler)

class PropDict:
    """
    Makes the function evaluation opaque. The function definition
    decides how to return the value rather then user.

    >>> f = PropDict(dict(n = 1, np1 = KWProp(lambda s : s.n + 1)))
    >>> f.np1
    2
    """
    post_process = process_kwprop
    def __init__(self, attrs):
        self.attrs = attrs

    def get(self, attr):
        return self.post_process(self.attrs[attr])

    def __getattr__(self, attr):
        try:
            return self.get(attr)
        except KeyError as e:
            raise AttributeError(e)

def wrap_funcs(func, wrapper):
    return (wrapper(func)
            if callable(func)
            else wrapper(func.fget)
            if KWProp.isinstance(func)
            else func)

def undo_wrap_funcs(func, wrapper):
    return  (KWProp(wrapper)
             if KWProp.isinstance(func)
             else wrapper)

def identity(func):
    return func

def wrap_key_to_attr_error(func):
    try:
        return func()
    except KeyError as e:
        raise AttributeError(e)

def hassignature(v):
    try:
        return inspect.signature(v)
    except ValueError as e:
        return False

def callablewithsignature(v):
    return callable(v) and hassignature(v)

class KWAsAttr:
    """
    Lets you access the default kwargs of function as it's attributes.

    >>> def two(one = 1): return one + 1
    >>> KWAsAttr(two).one
    1
    """
    def __init__(self, func):
        self._partial = dict()
        functools.update_wrapper(self, func)

    def _wrap_child(self, v):
        return type(self)(func = v) if callablewithsignature(v) else v

    def _wrap_kwdefaults(self, kwd):
        return { k : self._wrap_child(v) for k, v in kwd.items() }

    @property
    def partial(self):
        self._partial = (
            self._partial or self._wrap_kwdefaults(
                func_kwonlydefaults(self.__wrapped__)))
        return self._partial

    def __call__(self, *args, **kw):
        return self.__wrapped__(*args, **dict(self.partial, **kw))


    def __getattr__(self, k):
        try:
            return getattr(self.__wrapped__, k)
        except AttributeError as e:
            return wrap_key_to_attr_error(lambda : self.partial[k])

    def __setattr__(self, k, v):
        if (k in """func _func_defaults_ _wr_kw _partial _post_process
                    _undo_post_process""".split() or
            k.startswith("__") and k.endswith("__")):
            object.__setattr__(self, k, v)
        else:
            self.partial[k] = v

    def __repr__(self):
        if self._partial:
            return " ".join("""
            {self.__class__.__name__}( {self.__wrapped__}, {self.partial} )
            """.split()).format(self=self)
        else:
            return self.__class__.__name__ + "(" + repr(self.__wrapped__) + ")"


def extended_kwprop(func):
    """
    >>> p = KWProp
    >>> two = lambda one = 1: one + 1
    >>> two()
    2
    >>> ptwo = functools.partial(two, one = p(lambda s : s.zero + 1))
    >>> two2 = extended_kwprop(ptwo)
    >>> two2(zero = 10)
    12
    """
    attrs = func_kwonlydefaults(func)
    @functools.wraps(func)
    def wrapper(*args, **kw):
        argnames = nondefault_argnames(func)
        namedargs = dict(zip(argnames, args))
        kw.update(namedargs)
        kwprops_m = PropDict(dict(attrs, **kw))
        return apply_conf(func, kwprops_m, chain(argnames, attrs.keys()))

    return wrapper

def kwasattr_to_propdict(kwasattr):
    return PropDict(kwasattr.partial)

def xargs_(func, expect_args=(), *args, **kwargs):
    pfunc = functools.partial(func, *args, **kwargs)
    @functools.wraps(pfunc)
    def wrapper(conf):
        return apply_conf(pfunc, conf, expect_args)

    return wrapper

def xargs(func, expect_args=(), *args, **kwargs):
    return KWProp(xargs_(func, expect_args, *args, **kwargs))

def xargmem(func, expect_args=(), *args, **kwargs):
    return KWProp(method_memoizer(xargs_(func, expect_args, *args, **kwargs)))

def xargspartial(func, expect_args=(), *args, **kwargs):
    return KWProp(xargs_(functools.partial(functools.partial, func),
                         expect_args, *args, **kwargs))
    
def kwasattr_to_key_default(kwasattr):
    key_default = []
    for k, v in kwasattr.partial.items():
        if isinstance(v, type(kwasattr)):
            key_default.extend(
                [ ([k] + k1, v1) for k1, v1 in kwasattr_to_key_default(v) ])
        else:
            key_default.append(([k], v))
    return key_default

def stringify_key_defaults(key_default, sep="."):
    return [(sep.join(k), v) for k, v in key_default]

def split_key_values(key_values, sep="."):
    return [(k.split(sep), v) for k, v in key_values]

def parse_bool(s):
    return False if s == 'False' else bool(s)

def type_parser(t, glbls={}):
    if issubclass(t, bool):
        return parse_bool
    elif issubclass(t, (float, int, str)):
        return t
    elif issubclass(t, (dict, list)):
        return json.loads
    elif glbls: 
        return lambda s : glbls[s]
    else:
        raise ValueError("Do not know how to parse from string for type " + str(t))

@extended_kwprop
def update_argparser(kwasattr,
                     parser                   = KWProp(ArgumentParser),
                     key_default              = xargs(
                         kwasattr_to_key_default, ["kwasattr"]),
                     stringified_key_defaults = xargs(
                         stringify_key_defaults, ["key_default"])
):
    for k, deflt in stringified_key_defaults:
        try:
            parser.add_argument("--" + k, default=deflt,
                                type=type_parser(type(deflt)))
        except ValueError as e:
            pass
    return parser

def rec_setattr(obj, keys, val):
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], val)
        
@extended_kwprop
def update_kwasattr_from_argparse(kwasattr, 
                                  args,
                                  split_key_values = KWProp(
                                      lambda s: split_key_values(
                                          vars(s.args).items()))
):
    for k, v in split_key_values:
        rec_setattr(kwasattr, k, v)
    return kwasattr
    

@extended_kwprop
def parse_args_update_kwasattr(kwasattr,
                               argv = KWProp(
                                   lambda _ : sys.argv[1:]),
                               prep_parser = xargs(
                                   update_argparser,
                                   ["kwasattr"]),
                               args = KWProp(lambda s: s.prep_parser.parse_args(s.argv)),
                               updated_kwasattr = xargs(
                                   update_kwasattr_from_argparse,
                                   ["kwasattr", "args"])):
    return updated_kwasattr
                               
