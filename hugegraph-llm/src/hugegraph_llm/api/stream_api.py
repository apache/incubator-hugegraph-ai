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

from fastapi import status, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphRAGRequest,
)
from hugegraph_llm.config import prompt
from hugegraph_llm.utils.log import log


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
                        custom_related_information=req.custom_priority_info,
                        answer_prompt=req.answer_prompt or prompt.answer_prompt,
                        keywords_extract_prompt=req.keywords_extract_prompt
                                                or prompt.keywords_extract_prompt,
                        gremlin_tmpl_num=req.gremlin_tmpl_num,
                        gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                ):
                    # Format as Server-Sent Events
                    data = json.dumps({
                        "query": req.query,
                        "chunk": chunk
                    })
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            except Exception as e:
                log.error(f"Error in streaming RAG response: {e}")
                error_data = json.dumps({"error": str(e)})
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

        async def generate_graph_stream():
            try:
                async for chunk in graph_rag_recall_stream_func(
                        query=req.query,
                        gremlin_tmpl_num=req.gremlin_tmpl_num,
                        rerank_method=req.rerank_method,
                        near_neighbor_first=req.near_neighbor_first,
                        custom_related_information=req.custom_priority_info,
                        gremlin_prompt=req.gremlin_prompt or
                                       prompt.gremlin_generate_prompt,
                ):
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
                log.error(f"TypeError in streaming graph RAG recall: {e}")
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
            except Exception as e:
                log.error(f"Unexpected error in streaming graph RAG recall: {e}")
                error_data = json.dumps({"error": "An unexpected error occurred."})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            generate_graph_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
