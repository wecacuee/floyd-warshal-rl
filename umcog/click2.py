import functools
import inspect
import sys
import argparse


def func_required_args_from_sig(func):
    return [k
            for k, p in inspect.signature(func).parameters.items()
            if p.default is inspect._empty]

def func_kwonlydefaults_from_sig(func):
    return {k : p.default
            for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect._empty}


def func_kwonlydefaults(func):
    if isinstance(func, functools.partial):
        if func.args:
            reqargs = func_kwonlydefaults_from_sig(func.func)
            kw = zip(reqargs, func.args)
        else:
            kw = dict()
        kw.update(func_kwonlydefaults(func.func))
        kw.update(func.keywords)
    else:
        kw = func_kwonlydefaults_from_sig(func)
    return kw


def argparse_req_defaults(k):
    return dict(option_strings = ("{}".format(k),))


def argparse_opt_defaults(k, default):
    return dict(option_strings = ('--{}'.format(k),),
                type = type(default),
                default = default)


def foreach_argument(parser, defaults):
      option_strings = defaults.pop('option_strings')
      parser.add_argument(*option_strings, **defaults)


def add_argument_args_from_func_sig(func):
    """
    >>> def main(x, a = 1, b = 2, c = "C"):
    ...     return dict(x = x, a = a, b = b, c = c)
    >>> add_argument_args_from_func_sig(main)
    [{'option_strings': ('x',)}, {'option_strings': ('--a',), 'type': <class 'int'>, 'default': 1}, {'option_strings': ('--b',), 'type': <class 'int'>, 'default': 2}, {'option_strings': ('--c',), 'type': <class 'str'>, 'default': 'C'}]
    """
    parser_add_argument_args = []
    required_args = func_required_args_from_sig(func)
    for k in required_args:
        defaults = argparse_req_defaults(k)
        parser_add_argument_args.append(defaults)

    kwdefaults = func_kwonlydefaults(func)
    for k, deflt in kwdefaults.items():
        defaults = argparse_opt_defaults(k, deflt)
        parser_add_argument_args.append(defaults)
    return parser_add_argument_args


def argparser_from_func_sig(func,
                            argparseopts = dict(),
                            parser_factory = argparse.ArgumentParser,
                            foreach_argument_cb = foreach_argument):
    """
    """
    parser = parser_factory()
    for kw in add_argument_args_from_func_sig(func):
        foreach_argument_cb(parser, dict(kw, **argparseopts.get(k, dict())))
    return parser


def command(func,
            parser_factory = argparser_from_func_sig,
            sys_args_gen = lambda : sys.argv[1:]):
    """
    >>> @command
    ... def main(x, a = 1, b = 2, c = "C"):
    ...     return dict(x = x, a = a, b = b, c = c)
    >>> main(sys_args = ["X"])
    {'x': 'X', 'a': 1, 'b': 2, 'c': 'C'}
    >>> main(sys_args = "Y --a 2 --c D".split())
    {'x': 'Y', 'a': 2, 'b': 2, 'c': 'D'}
    """
    parser = parser_factory(func)
    @functools.wraps(func)
    def wrapper(sys_args = None, *args, **kw):
        # parse arguments when the function is actually called
        parsed_args = parser.parse_args(sys_args_gen()
                                        if sys_args is None
                                        else sys_args)
        return functools.partial(func, **vars(parsed_args))(*args, **kw)

    return wrapper

