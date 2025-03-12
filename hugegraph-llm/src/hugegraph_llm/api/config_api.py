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
from hugegraph_llm.api.models.rag_requests import (
    GraphConfigRequest,
    LLMConfigRequest,
    RerankerConfigRequest,
)
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import llm_settings


async def graph_config_route(router: APIRouter, apply_graph_conf):
    @router.post("/config/graph", status_code=status.HTTP_201_CREATED)
    async def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        res = await apply_graph_conf(req.ip, req.port, req.name, req.user, req.pwd, req.gs, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))
    return graph_config_api

async def llm_config_route(router: APIRouter, apply_llm_conf):
    # TODO: restructure the implement of llm to three types, like "/config/chat_llm" + /config/mini_task_llm + ..
    @router.post("/config/llm", status_code=status.HTTP_201_CREATED)
    async def llm_config_api(req: LLMConfigRequest):
        llm_settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            res = await apply_llm_conf(req.api_key, req.api_base, req.language_model, req.max_tokens,
                                       origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            res = await apply_llm_conf(req.api_key, req.secret_key, req.language_model, None, origin_call="http")
        else:
            res = await apply_llm_conf(req.host, req.port, req.language_model, None, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    return llm_config_api

async def embedding_config_route(router: APIRouter, apply_embedding_conf):
    @router.post("/config/embedding", status_code=status.HTTP_201_CREATED)
    async def embedding_config_api(req: LLMConfigRequest):
        llm_settings.embedding_type = req.llm_type

        if req.llm_type == "openai":
            res = await apply_embedding_conf(req.api_key, req.api_base, req.language_model, origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            res = await apply_embedding_conf(req.api_key, req.api_base, None, origin_call="http")
        else:
            res = await apply_embedding_conf(req.host, req.port, req.language_model, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    return embedding_config_api

async def rerank_config_route(router: APIRouter, apply_reranker_conf):
    @router.post("/config/rerank", status_code=status.HTTP_201_CREATED)
    async def rerank_config_api(req: RerankerConfigRequest):
        llm_settings.reranker_type = req.reranker_type

        if req.reranker_type == "cohere":
            res = await apply_reranker_conf(req.api_key, req.reranker_model, req.cohere_base_url, origin_call="http")
        elif req.reranker_type == "siliconflow":
            res = await apply_reranker_conf(req.api_key, req.reranker_model, None, origin_call="http")
        else:
            res = status.HTTP_501_NOT_IMPLEMENTED
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    return rerank_config_api


async def config_http_api(
    router: APIRouter,
    apply_graph_conf,
    apply_llm_conf,
    apply_embedding_conf,
    apply_reranker_conf,
):
    await graph_config_route(router, apply_graph_conf)
    await llm_config_route(router, apply_llm_conf)
    await embedding_config_route(router, apply_embedding_conf)
    await rerank_config_route(router, apply_reranker_conf)
