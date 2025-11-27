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

import os

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import LogStreamRequest
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import admin_settings


# FIXME: line 31: E0702: Raising dict while only classes or instances are allowed (raising-bad-type)
def admin_http_api(router: APIRouter, log_stream):
    @router.post("/logs", status_code=status.HTTP_200_OK)
    async def log_stream_api(req: LogStreamRequest):
        if admin_settings.admin_token != req.admin_token:
            raise generate_response(  # pylint: disable=raising-bad-type
                RAGResponse(
                    status_code=status.HTTP_403_FORBIDDEN,  # pylint: disable=E0702
                    message="Invalid admin_token",
                )
            )
        log_path = os.path.join("logs", req.log_file)

        # Create a StreamingResponse that reads from the log stream generator
        return StreamingResponse(log_stream(log_path), media_type="text/plain")
