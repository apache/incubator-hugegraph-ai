import functools
import threading


class SingletonBase(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class HGraphRouter(metaclass=SingletonBase):
    def __init__(self):
        self._routers = {}

    @staticmethod
    def get(method, uri):

        def decorator(func):

            @functools.wraps(func)
            def wrapper(cls, *args, **kwargs):
                func_params = func.__code__.co_varnames[: func.__code__.co_argcount]
                func_args = dict(zip(func_params[1:], args))
                all_kwargs = {**kwargs, **func_args}
                uri = uri.format(**all_kwargs)
                response = cls._sess.get(uri, **kwargs)
                func(cls, *args, **kwargs, __callback__=response)

            return wrapper

        return decorator

    def put(self):
        pass

    def post(self):
        pass

    def delete(self):
        pass


router = HGraphRouter()
