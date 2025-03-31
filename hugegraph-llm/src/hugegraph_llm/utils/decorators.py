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

import time
import asyncio
from functools import wraps
from typing import Optional, Any, Callable

from hugegraph_llm.utils.log import log


def log_elapsed_time(start_time: float, func: Callable, args: tuple, msg: Optional[str]):
    elapse_time = time.perf_counter() - start_time
    unit = "s"
    if elapse_time < 1:
        elapse_time *= 1000
        unit = "ms"

    class_name = args[0].__class__.__name__ if args else ""
    message = f"{class_name} {msg or f'func {func.__name__}()'}"
    log.info("%s took %.2f %s", message, elapse_time, unit)


def log_time(msg: Optional[str] = "") -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            log_elapsed_time(start_time, func, args, msg)
            return result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            log_elapsed_time(start_time, func, args, msg)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    # handle "@log_time" usage -> better to use "@log_time()" instead
    if callable(msg):
        return decorator(msg)
    return decorator


def log_operator_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        operator = args[1]
        log.debug("Running operator: %s", operator.__class__.__name__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        op_time = time.perf_counter() - start
        # Only record time ≥ 0.01s (10ms)
        if op_time >= 0.01:
            log.debug("Operator %s finished in %.2f seconds", operator.__class__.__name__, op_time)
            # log.debug("Current context:\n%s", result)
        return result
    return wrapper


def record_qps(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        call_count = result.get("call_count", 0)
        qps = call_count / (time.perf_counter() - start)
        if qps >= 0.10:
            log.debug("%s QPS: %.2f/s", args[0].__class__.__name__, qps)
        else:
            log.debug("%s QPS: %f/s", args[0].__class__.__name__, qps)
        return result
    return wrapper

def with_task_id(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import uuid
        task_id = f"task_{str(uuid.uuid4())}"
        log.info("New task created with id: %s", task_id)
        
        # Store the original return value
        result = func(*args, **kwargs)
        
        # Add the task_id to the function's context
        if hasattr(result, "__closure__") and result.__closure__:
            # If it's a closure, we can add the task_id to its context
            setattr(result, "task_id", task_id)
        
        return result
    return wrapper
