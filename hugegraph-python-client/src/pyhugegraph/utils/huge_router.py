import functools
import threading

from pyhugegraph.api.common import HugeParamsBase


class SingletonBase(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class HGraphRouterManager(metaclass=SingletonBase):
    def __init__(self):
        self._routers = []

    def add_router(self, uri):
        self._routers.append(uri)


class HGraphRouter:

    @staticmethod
    def get(uri):
        HGraphRouterManager().add_router(uri)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self: HugeParamsBase, *args, **kwargs):
                func_params = func.__code__.co_varnames[: func.__code__.co_argcount]
                func_args = dict(zip(func_params[1:], args))
                all_kwargs = {**kwargs, **func_args}
                formatted_uri = uri.format(**all_kwargs)
                callback = lambda *inner_args, **inner_kwargs: self._sess.get(
                    formatted_uri, *inner_args, **inner_kwargs
                )
                return func(self, *args, __callback__=callback, **kwargs)

            return decorator

        return decorator

    @staticmethod
    def put(uri):
        pass

    @staticmethod
    def post(uri):
        pass

    @staticmethod
    def delete(uri):
        pass
