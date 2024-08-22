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

from fastapi import status, APIRouter

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import RAGRequest, GraphConfigRequest, LLMConfigRequest
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import settings


def rag_http_api(router: APIRouter, rag_answer_func, apply_graph_conf, apply_llm_conf, apply_embedding_conf):
    @router.post("/rag", status_code=status.HTTP_200_OK)
    def rag_answer_api(req: RAGRequest):
        result = rag_answer_func(req.query, req.raw_llm, req.vector_only, req.graph_only, req.graph_vector)
        return {
            key: value
            for key, value in zip(["raw_llm", "vector_only", "graph_only", "graph_vector"], result)
            if getattr(req, key)
        }

    @router.post("/config/graph", status_code=status.HTTP_201_CREATED)
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        res = apply_graph_conf(req.ip, req.port, req.name, req.user, req.pwd, req.gs, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/llm", status_code=status.HTTP_201_CREATED)
    def llm_config_api(req: LLMConfigRequest):
        settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_llm_conf(
                req.api_key, req.api_base, req.language_model, req.max_tokens, origin_call="http"
            )
        elif req.llm_type == "qianfan_wenxin":
            res = apply_llm_conf(req.api_key, req.secret_key, req.language_model, None, origin_call="http")
        else:
            res = apply_llm_conf(req.host, req.port, req.language_model, None, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/embedding", status_code=status.HTTP_201_CREATED)
    def embedding_config_api(req: LLMConfigRequest):
        settings.embedding_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_embedding_conf(req.api_key, req.api_base, req.language_model, origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            res = apply_embedding_conf(req.api_key, req.api_base, None, origin_call="http")
        else:
            res = apply_embedding_conf(req.host, req.port, req.language_model, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))
