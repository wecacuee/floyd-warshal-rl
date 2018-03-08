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

class MethodMemoizer(object):
    def __init__(
            self,
            memoize_mem_attr = "_memoize_mem",
            keyfunc = lambda f, a, k: (f.__name__, a, tuple(sorted(k.items())))
    ):
        self.memoize_mem_attr = memoize_mem_attr
        self.keyfunc          = keyfunc

    def init_obj(self, s):
        setattr(s, self.memoize_mem_attr, dict())

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(s, *args, **kwargs):
            key = self.keyfunc(method, args, kwargs)
            try:
                memoize_mem = getattr(s, self.memoize_mem_attr)
            except AttributeError as a:
                self.init_obj(s)
                memoize_mem = getattr(s, self.memoize_mem_attr)

            if key not in memoize_mem:
                memoize_mem[key] =  method(s, *args, **kwargs)

            return memoize_mem[key]
        return wrapper

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
        def __init__(self):
            MEMOIZE_METHOD.init_obj(self)

        @MEMOIZE_METHOD
        def rand(self):
            return random.randint(0, 10000)
    a = A()
    r1 = a.rand()
    for i in range(10):
        print(f"{i} = {a.rand()}")
        assert r1 == a.rand(), "All vaues should be same"
