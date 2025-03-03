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

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import (
    RAGRequest,
    GraphConfigRequest,
    LLMConfigRequest,
    RerankerConfigRequest,
    GraphRAGRequest,
)
from hugegraph_llm.config import huge_settings
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import llm_settings, prompt
from hugegraph_llm.utils.log import log

# pylint: disable=too-many-statements
def rag_http_api(
    router: APIRouter,
    rag_answer_func,
    graph_rag_recall_func,
    apply_graph_conf,
    apply_llm_conf,
    apply_embedding_conf,
    apply_reranker_conf,
):
    @router.post("/rag", status_code=status.HTTP_200_OK)
    def rag_answer_api(req: RAGRequest):
        set_graph_config(req)

        result = rag_answer_func(
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

    def set_graph_config(req):
        if req.client_config:
            huge_settings.graph_ip = req.client_config.ip
            huge_settings.graph_port = req.client_config.port
            huge_settings.graph_name = req.client_config.name
            huge_settings.graph_user = req.client_config.user
            huge_settings.graph_pwd = req.client_config.pwd
            huge_settings.graph_space = req.client_config.gs

    @router.post("/rag/graph", status_code=status.HTTP_200_OK)
    def graph_rag_recall_api(req: GraphRAGRequest):
        try:
            set_graph_config(req)

            result = graph_rag_recall_func(
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
            )

            if req.get_vertex_only:
                from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
                graph_rag = GraphRAGQuery()
                graph_rag.init_client(result)
                vertex_details = graph_rag.get_vertex_details(result["match_vids"])

                if vertex_details:
                    result["match_vids"] = vertex_details

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
