import functools
import inspect
import sys
import argparse


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


def argparse_defaults(k, default):
    return dict(option_strings = ('--{}'.format(k),),
                type = type(default),
                default = default)


def command(func,
            argparseopts = dict(),
            parser_factory = argparse.ArgumentParser,
            description = lambda : sys.argv[0],
            sys_args_gen = lambda : sys.argv[1:]):
    """
    >>> @command
    ... def main(a = 1, b = 2, c = "C"):
    ...     return dict(a = a, b = b, c = c)
    >>> main(sys_args = [])
    {'a': 1, 'b': 2, 'c': 'C'}
    >>> main(sys_args = "--a 2 --c D".split())
    {'a': 2, 'b': 2, 'c': 'D'}
    """
    kwdefaults = func_kwonlydefaults(func)
    parser = parser_factory(description = description())
    for k, deflt in kwdefaults.items():
        defaults = argparse_defaults(k, deflt)
        defaults.update(argparseopts.get(k, dict()))
        option_strings = defaults.pop('option_strings')
        parser.add_argument(*option_strings, **defaults)

    @functools.wraps(func)
    def wrapper(sys_args = None, *args, **kw):
        # parse arguments when the function is actually called
        parsed_args = parser.parse_args(sys_args_gen()
                                        if sys_args is None
                                        else sys_args)
        return functools.partial(func, **vars(parsed_args))(*args, **kw)

    return wrapper

