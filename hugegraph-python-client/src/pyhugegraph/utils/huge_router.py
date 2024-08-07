# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
import json
import inspect
import functools
import threading


from typing import Any, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyhugegraph.api.common import HGraphContext


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
        self._routers = {}

    def add_router(self, key, uri):
        self._routers.update({key: uri})

    def get_routers(self):
        return self._routers

    def __repr__(self) -> str:
        return json.dumps(self._routers, indent=4)


def http(method: str, uri: str) -> Callable:
    """
    A decorator to format the URI and inject a request function into the decorated method.

    Args:
        uri (str): The URI template to be formatted with function arguments.

    Returns:
        Callable: The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        HGraphRouterManager().add_router(func.__qualname__, uri)

        @functools.wraps(func)
        def wrapper(self: "HGraphContext", *args: Any, **kwargs: Any) -> Any:
            if re.search(r"{\w+}", uri):
                sig = inspect.signature(func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                all_kwargs = dict(bound_args.arguments)
                all_kwargs.pop("self", None)
                formatted_uri = uri.format(**all_kwargs)
            else:
                formatted_uri = uri

            # Use functools.partial to create a partial function for making requests
            make_request = functools.partial(self._sess.request, formatted_uri, method)
            # Store the partial function on the instance
            setattr(self, f"_{func.__name__}_request", make_request)

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class HGraphRouter:
    def _http_request(self, name):
        return getattr(self, f"_{name}_request")()
