import functools
class Memoizer(object):
    def __init__(self, memoize_mem=None):
        self.memoize_mem = memoize_mem or dict()

    def __call__(self, func 
                 , keyfunc=lambda f, a, k: (f.__name__, a, tuple(sorted(k.items())))):
        """
        >>> memoize = Memoizer()
        >>> x = 10
        >>> incy = lambda y : x + y
        >>> incy(1)
        11
        >>> x = 11
        >>> incy(1)
        12
        >>> mincy = memoize(incy)
        >>> mincy(1)
        12
        >>> x = 10
        >>> mincy(1)
        12
        >>> incy(1)
        11
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = keyfunc(func, args, kwargs)
            if key not in self.memoize_mem:
                self.memoize_mem[key] =  func(*args, **kwargs)

            return self.memoize_mem[key]
        return wrapper

    def flush(self):
        self.memoize_mem = dict()
        

def defaultkeyfunc(method, args, kwargs, 
                    funckey    = lambda f     : id(f),
                    argskey    = lambda args  : args,
                    kwargskey  = lambda kwargs: tuple(sorted(kwargs.items()))):
    return funckey(method), argskey(args), kwargskey(kwargs)


def method_memoizer(method     = None,
                    memory_key = "_memoize_cache",
                    keyfunc    = defaultkeyfunc):

    @functools.wraps(method)
    def wrapper(s, *args, **kwargs):
        memory = getattr(s, memory_key, dict())
        setattr(s, memory_key, memory)
        key = keyfunc(method, args, kwargs)
        if key not in memory:
            return memory.setdefault(key, method(s, *args, **kwargs))
        else:
            return memory[key]
    return wrapper


class MethodMemoizer(object):
    def __init__(
            self,
            memoize_mem_attr = "_memoize_mem",
            f_key = lambda f: f.__name__,
            a_key = lambda a: a,
            kw_key = lambda kw: tuple(sorted(kw.items()))
    ):
        self.memoize_mem_attr = memoize_mem_attr
        self.f_key = f_key
        self.a_key = a_key
        self.kw_key = kw_key

    def keyfunc(self, f, a, kw):
        return (self.f_key(f), self.a_key(a), self.kw_key(kw))

    def memoize_with_keyfunc(self, method, keyfunc):
        return method_memoizer(method, keyfunc=keyfunc)

    def __call__(self, method):
        return self.memoize_with_keyfunc(method, self.keyfunc)

def LambdaMethodMemoizer(func_name, **kw):
    return MethodMemoizer(f_key = lambda f: func_name, **kw)

"""
Global object to memoize methods.
This object does not store anything. The memoize dictionary is
attached to the object. This just contains the attribute name that is
used to store the memoize data.
"""
MEMOIZE_METHOD = MethodMemoizer()

if __name__ == '__main__':
    MEMOIZE_METHOD = MethodMemoizer()
    import random
    class A:
        rand2 = LambdaMethodMemoizer("rand2")(
            lambda s: random.randint(0, 100))
        @MEMOIZE_METHOD
        def rand(self):
            return random.randint(0, 10000)
    a = A()
    r1 = a.rand()
    for i in range(10):
        print("{i} = {rand}".format(i=i,rand=a.rand()))
        assert r1 == a.rand(), "All vaues should be same"
    r2 = a.rand2()
    for i in range(10):
        print("{i} = {rand}".format(i=i,rand=a.rand2()))
        assert r2 == a.rand2(), "All vaues should be same"
