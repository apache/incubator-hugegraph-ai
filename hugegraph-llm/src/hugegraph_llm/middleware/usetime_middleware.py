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

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
from hugegraph_llm.utils.log import log


class UseTimeMiddleware(BaseHTTPMiddleware):
    """Middleware to add process time to response headers"""
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # TODO: handle time record for async task pool in gradio
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000 # ms
        unit = "ms"
        if process_time > 1000:
            process_time /= 1000
            unit = "s"

        response.headers["X-Process-Time"] = f"{process_time:.2f} {unit}"
        log.info("Request process time: %.2f ms, code=%d", process_time, response.status_code)
        log.info(f"{request.method} - Args: {request.query_params}, IP: {request.client.host}, URL: {request.url}")
        return response
