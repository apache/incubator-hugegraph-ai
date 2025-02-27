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


from fastapi import status, APIRouter, HTTPException
from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphRAGRequest,
)
from hugegraph_llm.config import prompt
from hugegraph_llm.utils.log import log


async def rag_http_api(
        router: APIRouter,
        rag_answer_func,
        graph_rag_recall_func,
):
    @router.post("/rag", status_code=status.HTTP_200_OK)
    async def rag_answer_api(req: RAGRequest):
        result = await rag_answer_func(
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
        )

        return {
            "query": req.query,
            **{
                key: value
                for key, value in zip(["raw_answer", "vector_only", "graph_only",
                                       "graph_vector_answer"], result)
                if getattr(req, key)
            },
        }

    @router.post("/rag/graph", status_code=status.HTTP_200_OK)
    async def graph_rag_recall_api(req: GraphRAGRequest):
        try:
            result = await graph_rag_recall_func(
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

            return {"graph_recall": json.dumps(result)}

        except TypeError as e:
            log.error("TypeError in graph_rag_recall_api: %s", e)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except Exception as e:
            log.error("Unexpected error occurred: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred."
            ) from e
