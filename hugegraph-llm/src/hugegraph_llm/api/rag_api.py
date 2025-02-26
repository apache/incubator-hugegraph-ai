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

import json
import asyncio
from typing import AsyncGenerator

from fastapi import status, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphConfigRequest,
    LLMConfigRequest,
    RerankerConfigRequest,
    GraphRAGRequest,
)
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import llm_settings, prompt
from hugegraph_llm.utils.log import log


def rag_http_api(
        router: APIRouter,
        rag_answer_func,
        graph_rag_recall_func,
        apply_graph_conf,
        apply_llm_conf,
        apply_embedding_conf,
        apply_reranker_conf,
        rag_answer_stream_func=None,
        graph_rag_recall_stream_func=None,
):
    async def stream_rag_answer(
            text,
            raw_answer,
            vector_only_answer,
            graph_only_answer,
            graph_vector_answer,
            graph_ratio,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            answer_prompt,
            keywords_extract_prompt,
            gremlin_tmpl_num,
            gremlin_prompt,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the RAG answer results
        """
        if rag_answer_stream_func:
            # If a streaming-specific function exists, use it
            async for chunk in rag_answer_stream_func(
                    text=text,
                    raw_answer=raw_answer,
                    vector_only_answer=vector_only_answer,
                    graph_only_answer=graph_only_answer,
                    graph_vector_answer=graph_vector_answer,
                    graph_ratio=graph_ratio,
                    rerank_method=rerank_method,
                    near_neighbor_first=near_neighbor_first,
                    custom_related_information=custom_related_information,
                    answer_prompt=answer_prompt,
                    keywords_extract_prompt=keywords_extract_prompt,
                    gremlin_tmpl_num=gremlin_tmpl_num,
                    gremlin_prompt=gremlin_prompt,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            # Otherwise, use the normal function but adapt it for streaming
            # by sending the entire result at once
            result = rag_answer_func(
                text=text,
                raw_answer=raw_answer,
                vector_only_answer=vector_only_answer,
                graph_only_answer=graph_only_answer,
                graph_vector_answer=graph_vector_answer,
                graph_ratio=graph_ratio,
                rerank_method=rerank_method,
                near_neighbor_first=near_neighbor_first,
                custom_related_information=custom_related_information,
                answer_prompt=answer_prompt,
                keywords_extract_prompt=keywords_extract_prompt,
                gremlin_tmpl_num=gremlin_tmpl_num,
                gremlin_prompt=gremlin_prompt,
            )

            response_data = {
                "query": text,
                **{
                    key: value
                    for key, value in zip(["raw_answer", "vector_only", "graph_only", "graph_vector_answer"], result)
                    if eval(key)  # Convert string to boolean
                },
            }

            yield f"data: {json.dumps(response_data)}\n\n"
            # Signal end of stream
            yield "data: [DONE]\n\n"

    async def stream_graph_rag_recall(
            query,
            gremlin_tmpl_num,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            gremlin_prompt,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the graph RAG recall results
        """
        if graph_rag_recall_stream_func:
            # If a streaming-specific function exists, use it
            async for chunk in graph_rag_recall_stream_func(
                    query=query,
                    gremlin_tmpl_num=gremlin_tmpl_num,
                    rerank_method=rerank_method,
                    near_neighbor_first=near_neighbor_first,
                    custom_related_information=custom_related_information,
                    gremlin_prompt=gremlin_prompt,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            # Otherwise, use the normal function but adapt it for streaming
            try:
                result = graph_rag_recall_func(
                    query=query,
                    gremlin_tmpl_num=gremlin_tmpl_num,
                    rerank_method=rerank_method,
                    near_neighbor_first=near_neighbor_first,
                    custom_related_information=custom_related_information,
                    gremlin_prompt=gremlin_prompt,
                )

                if isinstance(result, dict):
                    params = [
                        "query",
                        "keywords",
                        "match_vids",
                        "graph_result_flag",
                        "gremlin",
                        "graph_result",
                        "vertex_degree_list",
                    ]
                    user_result = {key: result[key] for key in params if key in result}
                    yield f"data: {json.dumps({'graph_recall': user_result})}\n\n"
                else:
                    # Note: Maybe only for qianfan/wenxin
                    yield f"data: {json.dumps({'graph_recall': json.dumps(result)})}\n\n"

                # Signal end of stream
                yield "data: [DONE]\n\n"

            except TypeError as e:
                log.error("TypeError in stream_graph_rag_recall: %s", e)
                yield f"data: {json.dumps({'error': str(e), 'status': 400})}\n\n"
            except Exception as e:
                log.error("Unexpected error occurred: %s", e)
                yield f"data: {json.dumps({'error': 'An unexpected error occurred.', 'status': 500})}\n\n"

    @router.post("/rag", status_code=status.HTTP_200_OK)
    async def rag_answer_api(req: RAGRequest):
        if req.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_rag_answer(
                    text=req.query,
                    raw_answer=req.raw_answer,
                    vector_only_answer=req.vector_only,
                    graph_only_answer=req.graph_only,
                    graph_vector_answer=req.graph_vector_answer,
                    graph_ratio=req.graph_ratio,
                    rerank_method=req.rerank_method,
                    near_neighbor_first=req.near_neighbor_first,
                    custom_related_information=req.custom_priority_info,
                    answer_prompt=req.answer_prompt or prompt.answer_prompt,
                    keywords_extract_prompt=req.keywords_extract_prompt or prompt.keywords_extract_prompt,
                    gremlin_tmpl_num=req.gremlin_tmpl_num,
                    gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                ),
                media_type="text/event-stream",
            )
        else:
            # Synchronous response (original behavior)
            result = rag_answer_func(
                text=req.query,
                raw_answer=req.raw_answer,
                vector_only_answer=req.vector_only,
                graph_only_answer=req.graph_only,
                graph_vector_answer=req.graph_vector_answer,
                graph_ratio=req.graph_ratio,
                rerank_method=req.rerank_method,
                near_neighbor_first=req.near_neighbor_first,
                custom_related_information=req.custom_priority_info,
                answer_prompt=req.answer_prompt or prompt.answer_prompt,
                keywords_extract_prompt=req.keywords_extract_prompt or prompt.keywords_extract_prompt,
                gremlin_tmpl_num=req.gremlin_tmpl_num,
                gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
            )
            # TODO: we need more info in the response for users to understand the query logic
            return {
                "query": req.query,
                **{
                    key: value
                    for key, value in zip(["raw_answer", "vector_only", "graph_only", "graph_vector_answer"], result)
                    if getattr(req, key)
                },
            }

    @router.post("/rag/graph", status_code=status.HTTP_200_OK)
    async def graph_rag_recall_api(req: GraphRAGRequest):
        if req.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_graph_rag_recall(
                    query=req.query,
                    gremlin_tmpl_num=req.gremlin_tmpl_num,
                    rerank_method=req.rerank_method,
                    near_neighbor_first=req.near_neighbor_first,
                    custom_related_information=req.custom_priority_info,
                    gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                ),
                media_type="text/event-stream",
            )
        else:
            # Synchronous response (original behavior)
            try:
                result = graph_rag_recall_func(
                    query=req.query,
                    gremlin_tmpl_num=req.gremlin_tmpl_num,
                    rerank_method=req.rerank_method,
                    near_neighbor_first=req.near_neighbor_first,
                    custom_related_information=req.custom_priority_info,
                    gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                )

                if isinstance(result, dict):
                    params = [
                        "query",
                        "keywords",
                        "match_vids",
                        "graph_result_flag",
                        "gremlin",
                        "graph_result",
                        "vertex_degree_list",
                    ]
                    user_result = {key: result[key] for key in params if key in result}
                    return {"graph_recall": user_result}
                # Note: Maybe only for qianfan/wenxin
                return {"graph_recall": json.dumps(result)}

            except TypeError as e:
                log.error("TypeError in graph_rag_recall_api: %s", e)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
            except Exception as e:
                log.error("Unexpected error occurred: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred."
                ) from e

    @router.post("/config/graph", status_code=status.HTTP_201_CREATED)
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        res = apply_graph_conf(req.ip, req.port, req.name, req.user, req.pwd, req.gs, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    # TODO: restructure the implement of llm to three types, like "/config/chat_llm"
    @router.post("/config/llm", status_code=status.HTTP_201_CREATED)
    def llm_config_api(req: LLMConfigRequest):
        llm_settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_llm_conf(req.api_key, req.api_base, req.language_model, req.max_tokens, origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            res = apply_llm_conf(req.api_key, req.secret_key, req.language_model, None, origin_call="http")
        else:
            res = apply_llm_conf(req.host, req.port, req.language_model, None, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/embedding", status_code=status.HTTP_201_CREATED)
    def embedding_config_api(req: LLMConfigRequest):
        llm_settings.embedding_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_embedding_conf(req.api_key, req.api_base, req.language_model, origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            res = apply_embedding_conf(req.api_key, req.api_base, None, origin_call="http")
        else:
            res = apply_embedding_conf(req.host, req.port, req.language_model, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/rerank", status_code=status.HTTP_201_CREATED)
    def rerank_config_api(req: RerankerConfigRequest):
        llm_settings.reranker_type = req.reranker_type

        if req.reranker_type == "cohere":
            res = apply_reranker_conf(req.api_key, req.reranker_model, req.cohere_base_url, origin_call="http")
        elif req.reranker_type == "siliconflow":
            res = apply_reranker_conf(req.api_key, req.reranker_model, None, origin_call="http")
        else:
            res = status.HTTP_501_NOT_IMPLEMENTED
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))