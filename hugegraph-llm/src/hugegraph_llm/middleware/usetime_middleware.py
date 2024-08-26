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
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        log.info("Process time: %s", process_time)
        log.info(f"Method: {request.method}, Args: {request.query_params}, IP: {request.client.host}, "
                 f"URL: {request.url}, Headers: {request.headers}")
        return response
