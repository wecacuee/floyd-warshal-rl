import importlib
from argparse import Namespace, ArgumentParser
import inspect

class Conf(Namespace):
    def items(self):
        return vars(self).items()

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    @classmethod
    def parser(cls):
        parser = ArgumentParser()
        parser.add_argument("--config", default="conf.default:CurrentConf")
        return parser

    @classmethod
    def import_class(cls, name):
        module_name, class_ = name.strip().split(":")
        module = importlib.import_module(module_name)
        return getattr(module, class_)

    @classmethod
    def parse_all_args(cls, args):
        c, remargs = cls.parser().parse_known_args(args)
        conf = cls.import_class(c.config)()
        return conf.parse_remargs(remargs)

    def parse_remargs(self, remargs):
        parser = self.parser()
        for k, v in vars(self).items():
            if isinstance(v, Conf):
                conf_from_args = lambda a : v.parse_remargs(a)
                parser.add_argument(
                    f"--{k}", default=None, type=conf_from_args)
            else:
                parser.add_argument(f"--{k}", default=None, type=type(v))
        args = parser.parse_args(remargs)
        self.__init__(
            **{ k : v for k, v in vars(args).items() if v is not None })
        self.run_checks()
        return self

    def run_checks(self):
        return True

def dict_update_recursive(ddest, dsrc):
    for k, v in dsrc.items():
        if hasattr(v, "items"):
            if k in ddest: 
                ddest[k] = dict_update_recursive(ddest[k], v)
            else:
                ddest[k] = v.copy()
        else:
            ddest[k] = v

    return ddest

def init_class_stack(classes, class_kwargs_list):
    # Initialize a stack of algorithm objects with their wrappers
    obj = None
    for class_, kwargs in zip(classes, class_kwargs_list):
        required_args = list(inspect.signature(class_).parameters.keys())
        if alg_obj is None:
            kw = { k : kwargs[k] for k in required_args }
            obj = class_(** kw )
        else:
            kw = { k : kwargs[k] for k in required_args[1:] }
            obj = class_(alg_obj, **kw)

    return obj
    

def NewConfClass(name, **kwargs):
    """
    Defines a new class with name = name and
    initialize kwargs attributes as property objects.
    All arguments to the kwargs should be callables (usually lambda functions).
    This is useful for redirecting configurations to same values
    ```
    MyConf = NewConfClass("MyConf",
                abc = lambda s : s.bleh.abc,
                xyz = lambda s : s._parent.blah.xyz)
    ```

    is equivalent to

    ```
    class MyConf(Conf):
        @property
        def abc(s):
            return s.bleh.abc

        @property
        def xyz(s):
            return s._parent.blah.xyz
    ```
    """
    for v in kwargs.values():
        assert callable(v), "Not implemented"
    return type(name, (Conf,),
                { k : property(v) if callable(v) else v
                  for k, v in kwargs.items() })
