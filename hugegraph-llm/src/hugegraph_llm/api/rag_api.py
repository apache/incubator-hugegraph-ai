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
from typing import Literal

from fastapi import status, APIRouter, HTTPException

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphConfigRequest,
    LLMConfigRequest,
    RerankerConfigRequest, GraphRAGRequest,
)
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import settings
from hugegraph_llm.utils.log import log


def graph_rag_recall(
        text: str,
        rerank_method: Literal["bleu", "reranker"],
        near_neighbor_first: bool,
        custom_related_information: str
) -> dict:
    from hugegraph_llm.operators.graph_rag_task import RAGPipeline
    rag = RAGPipeline()
    rag.extract_keywords().keywords_to_vid().query_graphdb().merge_dedup_rerank(
        rerank_method=rerank_method,
        near_neighbor_first=near_neighbor_first,
        custom_related_information=custom_related_information,
    )
    context = rag.run(verbose=True, query=text, graph_search=True)
    return context


def rag_http_api(
        router: APIRouter, rag_answer_func, apply_graph_conf, apply_llm_conf, apply_embedding_conf, apply_reranker_conf
):
    @router.post("/rag", status_code=status.HTTP_200_OK)
    def rag_answer_api(req: RAGRequest):
        result = rag_answer_func(req.query, req.raw_llm, req.vector_only, req.graph_only, req.graph_vector,
                                 req.answer_prompt)
        return {
            key: value
            for key, value in zip(["raw_llm", "vector_only", "graph_only", "graph_vector"], result)
            if getattr(req, key)
        }

    @router.post("/rag/graph", status_code=status.HTTP_200_OK)
    def graph_rag_recall_api(req: GraphRAGRequest):
        try:
            result = graph_rag_recall(
                text=req.query,
                rerank_method=req.rerank_method,
                near_neighbor_first=req.near_neighbor_first,
                custom_related_information=req.custom_related_information
            )
            # TODO/FIXME: handle QianFanClient error (not dict..critical)
            # log.critical(f"## {type(result)}, {json.dumps(result)}")
            if isinstance(result, dict):
                log.critical(f"##1. {type(result)}")
                return {"graph_recall": result}
            else:
                log.critical(f"##2. {type(result)}")
                return {"graph_recall": json.dumps(result)}

        except TypeError as e:
            log.error(f"TypeError in graph_rag_recall_api: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            log.error(f"Unexpected error occurred: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


    @router.post("/config/graph", status_code=status.HTTP_201_CREATED)
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        res = apply_graph_conf(req.ip, req.port, req.name, req.user, req.pwd, req.gs, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/llm", status_code=status.HTTP_201_CREATED)
    def llm_config_api(req: LLMConfigRequest):
        settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_llm_conf(req.api_key, req.api_base, req.language_model, req.max_tokens, origin_call="http")
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

    @router.post("/config/rerank", status_code=status.HTTP_201_CREATED)
    def rerank_config_api(req: RerankerConfigRequest):
        settings.reranker_type = req.reranker_type

        if req.reranker_type == "cohere":
            res = apply_reranker_conf(req.api_key, req.reranker_model, req.cohere_base_url, origin_call="http")
        elif req.reranker_type == "siliconflow":
            res = apply_reranker_conf(req.api_key, req.reranker_model, None, origin_call="http")
        else:
            res = status.HTTP_501_NOT_IMPLEMENTED
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))
