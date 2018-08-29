import operator
from functools import update_wrapper, partial, reduce
from itertools import tee, islice
from typing import TypeVar, Callable, Iterable, Mapping, Any, Tuple
from numbers import Number

T = TypeVar('T')

def slidingiter(iterable : Iterable, stride : int, size : int) -> Iterable[Tuple]:
    """
    >>> list(pairwise([1, 2, 3, 4, 5, 6]))
    [(1, 2), (3, 4), (5, 6)]

    >>> list(slidingiter([1, 2, 3, 4, 5], stride = 1, size = 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    iters = tee(iterable, size)
    return zip(*(islice(it, i, None, stride) for i, it in enumerate(iters)))


pairwise = partial(slidingiter, stride = 2, size = 2)
"""
s -> (s0,s1), (s2,s3), (s4, s5), ...
"""


def _apply_kw(ret, f):
    return f(**ret)


def _kwcompose(*fs,
               apply_one=_apply_kw,
               **kw):
    if len(fs) < 2:
        raise ValueError("Need at least one function to compose")
    return reduce(apply_one, fs[:-1], fs[-1](**kw))


def kwcompose(*a, **kw):
    return partial(_kwcompose, *a, **kw)


def _apply_rev(ret, f):
    return f(ret)

compose = partial(kwcompose, apply_one = _apply_rev)


def dictzip(kwiterables : Mapping[Any, Iterable]) -> Iterable[Mapping]:
    """
    >>> list(dictzip(dict(x = [2, 3], square = [4, 9])))
    [{'x': 2, 'square': 4}, {'x': 3, 'square': 9}]
    """
    keys, values = zip(*kwiterables.items())
    return (dict(zip(keys, v)) for v in zip(*values))


def kwmap(function, **kwiterables):
    return (function(**kw) for kw in dictzip(kwiterables))


def prod(seq : Iterable[Number]) -> Number:
    """
    >>> prod([1, 2, 3])
    6
    >>> prod([7, 7])
    49
    """
    seq = iter(seq)
    first = next(seq)
    return reduce(operator.mul, seq, first)


def identity(xyz: T) -> T:
    """
    >>> identity(1)
    1
    """
    return xyz


def const(xyz: T) -> Callable[[], T]:
    """
    >>> one = const(1)
    >>> one()
    1
    """
    return partial(identity, xyz)
