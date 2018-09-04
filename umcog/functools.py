import operator
from functools import update_wrapper, partial, reduce, wraps
from itertools import tee, islice
from typing import (TypeVar, Callable, Iterable, Mapping, Any, Tuple, Union,
                    Sequence, Iterator, MutableMapping)
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
    return reduce(apply_one, reversed(fs), **kw)


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


def list_reducer(acc, x):
    acc.append(x)
    return acc


def head(xs: Union[Iterable[T], Iterator[T]]) -> T:
    """
    >>> head([1, 2, 3])
    1
    >>> head(range(2, 10))
    2
    >>> head('abc')
    'a'
    """
    if not isinstance(xs, Iterator):
        xs = iter(xs)
    return next(xs)

first = head


def getitem(xs: Sequence, n):
    """
    >>> getitem([1, 2, 3], 1)
    2
    >>> getitem(range(2, 10), 0)
    2
    >>> getitem('abc', -1)
    'c'
    """
    return xs[n]

second = partial(getitem, n = 1)


def tail(xs: Union[Iterable[T], Iterator[T]]) -> Iterable[T]:
    """
    >>> list(tail([1, 2, 3]))
    [2, 3]
    >>> list(tail(range(2, 5)))
    [3, 4]
    >>> "".join(tail('abc'))
    'bc'
    """
    if not isinstance(xs, Iterator):
        xs = iter(xs)
    return islice(xs, 1, None)

rest = tail

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')


def groupby(iter_: Iterable[A],
            keyfun: Callable[[A], B] = head,
            valfun: Callable[[A], C]  = second,
            default_gen: Callable[[], D] = list,
            reducer: Callable[[D, C], D] = list_reducer,
            grouped_init: Callable[[], MutableMapping[B, D]] = dict):
    """
    >>> groupby([(1, 'a'), (2, 'b'), (1, 'aa'), (3, 'c')])
    {1: ['a', 'aa'], 2: ['b'], 3: ['c']}
    """
    grouped = grouped_init()
    for x in iter_:
        key = keyfun(x)
        acc = grouped.get(key, default_gen())
        grouped[key] = list_reducer(acc, valfun(x))
    return grouped


def _sum(*args):
    return sum(args)

sumby = partial(groupby, default_gen = const(0), reducer = _sum)


def _count(acc, _):
    return acc + 1

countby = partial(groupby, default_gen = const(0), reducer = _count)


def apply_unpkd(args, f):
    return f(*args)


def _unpkd_compose(fs,
                   *args,
                   apply_one = apply_unpkd):
    return reduce(apply_one, reversed(fs), args)


unpkd_compose = partial(partial, _unpkd_compose)


def revargs(func):
    @wraps(func)
    def wrapped(*args):
        return func(*reversed(args))
    return wrapped
