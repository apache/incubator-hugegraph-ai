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

import asyncio
import json

from fastapi import status, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphRAGRequest,
)
from hugegraph_llm.config import prompt, huge_settings
from hugegraph_llm.utils.log import log


# pylint: disable=too-many-statements
async def stream_http_api(
        router: APIRouter,
        rag_answer_stream_func,
        graph_rag_recall_stream_func,
):
    @router.post("/rag/stream", status_code=status.HTTP_200_OK)
    async def rag_answer_stream_api(req: RAGRequest):
        if not req.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Stream parameter must be set to True for streaming endpoint"
            )

        if req.client_config:
            huge_settings.graph_ip = req.client_config.ip
            huge_settings.graph_port = req.client_config.port
            huge_settings.graph_name = req.client_config.name
            huge_settings.graph_user = req.client_config.user
            huge_settings.graph_pwd = req.client_config.pwd
            huge_settings.graph_space = req.client_config.gs

        async def generate_stream():
            try:
                async for chunk in rag_answer_stream_func(
                        text=req.query,
                        raw_answer=req.raw_answer,
                        vector_only_answer=req.vector_only,
                        graph_only_answer=req.graph_only,
                        graph_vector_answer=req.graph_vector_answer,
                        graph_ratio=req.graph_ratio,
                        rerank_method=req.rerank_method,
                        near_neighbor_first=req.near_neighbor_first,
                        gremlin_tmpl_num=req.gremlin_tmpl_num,
                        max_graph_items=req.max_graph_items,
                        topk_return_results=req.topk_return_results,
                        vector_dis_threshold=req.vector_dis_threshold,
                        topk_per_keyword=req.topk_per_keyword,
                        # Keep prompt params in the end
                        custom_related_information=req.custom_priority_info,
                        answer_prompt=req.answer_prompt or prompt.answer_prompt,
                        keywords_extract_prompt=req.keywords_extract_prompt or prompt.keywords_extract_prompt,
                        gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                ):
                    # Format as Server-Sent Events
                    data = json.dumps({
                        "query": req.query,
                        "chunk": chunk
                    })
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            except (ValueError, TypeError) as e:  # More specific exceptions
                log.error("Error in streaming RAG response: %s", e)
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
            except Exception as e:  # pylint: disable=broad-exception-caught
                # We need to catch all exceptions here to ensure proper error response
                log.error("Unexpected error in streaming RAG response: %s", e)
                error_data = json.dumps({"error": "An unexpected error occurred"})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    @router.post("/rag/graph/stream", status_code=status.HTTP_200_OK)
    async def graph_rag_recall_stream_api(req: GraphRAGRequest):
        if not req.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Stream parameter must be set to True for streaming endpoint"
            )

        # Set graph config if provided
        if req.client_config:
            huge_settings.graph_ip = req.client_config.ip
            huge_settings.graph_port = req.client_config.port
            huge_settings.graph_name = req.client_config.name
            huge_settings.graph_user = req.client_config.user
            huge_settings.graph_pwd = req.client_config.pwd
            huge_settings.graph_space = req.client_config.gs

        async def generate_graph_stream():
            try:
                async for chunk in graph_rag_recall_stream_func(
                        query=req.query,
                        max_graph_items=req.max_graph_items,
                        topk_return_results=req.topk_return_results,
                        vector_dis_threshold=req.vector_dis_threshold,
                        topk_per_keyword=req.topk_per_keyword,
                        gremlin_tmpl_num=req.gremlin_tmpl_num,
                        rerank_method=req.rerank_method,
                        near_neighbor_first=req.near_neighbor_first,
                        custom_related_information=req.custom_priority_info,
                        gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                        get_vertex_only=req.get_vertex_only
                ):
                    # Handle vertex details for a get_vertex_only flag
                    if req.get_vertex_only and isinstance(chunk, dict) and "match_vids" in chunk:
                        from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
                        graph_rag = GraphRAGQuery()
                        graph_rag.init_client(chunk)
                        vertex_details = await graph_rag.get_vertex_details(chunk["match_vids"])
                        if vertex_details:
                            chunk["match_vids"] = vertex_details

                    if isinstance(chunk, dict):
                        params = [
                            "query",
                            "keywords",
                            "match_vids",
                            "graph_result_flag",
                            "gremlin",
                            "graph_result",
                            "vertex_degree_list",
                        ]
                        user_result = {key: chunk[key] for key in params if key in chunk}
                        data = json.dumps({"graph_recall": user_result})
                    else:
                        data = json.dumps({"graph_recall": json.dumps(chunk)})

                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)  # Small delay
            except TypeError as e:
                log.error("TypeError in streaming graph RAG recall: %s", e)
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
            except Exception as e:  # pylint: disable=broad-exception-caught
                # We need to catch all exceptions here to ensure proper error response
                log.error("Unexpected error in streaming graph RAG recall: %s", e)
                error_data = json.dumps({"error": "An unexpected error occurred"})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            generate_graph_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
