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

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from hugegraph_llm.config import settings


class RAGRequest(BaseModel):
    query: str
    raw_llm: Optional[bool] = True
    vector_only: Optional[bool] = False
    graph_only: Optional[bool] = False
    graph_vector: Optional[bool] = False


class GraphConfigRequest(BaseModel):
    ip: str = "127.0.0.1"
    port: str = "8080"
    name: str = "hugegraph"
    user: str = "xxx"
    pwd: str = "xxx"
    gs: str = None


class LLMConfigRequest(BaseModel):
    llm_type: str
    # The common parameters shared by OpenAI, Qianfan Wenxin,
    # and OLLAMA platforms.
    api_key: str
    api_base: str
    language_model: str
    # Openai-only properties
    max_tokens: str = None
    # qianfan-wenxin-only properties
    secret_key: str = None
    # ollama-only properties
    host: str = None
    port: str = None


def rag_http_api(app: FastAPI, rag_answer_func, apply_graph_conf, apply_llm_conf, apply_embedding_conf):
    @app.post("/rag")
    def rag_answer_api(req: RAGRequest):
        result = rag_answer_func(req.query, req.raw_llm, req.vector_only, req.graph_only, req.graph_vector)
        return {
            key: value
            for key, value in zip(["raw_llm", "vector_only", "graph_only", "graph_vector"], result)
            if getattr(req, key)
        }

    @app.post("/graph/config")
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        status_code = apply_graph_conf(req.ip, req.port, req.name, req.user, req.pwd, req.gs, origin_call="http")
        return {
            "message": (
                "Connection successful. Configured finished."
                if 200 <= status_code < 300
                else (
                    "Unsupported HTTP method"
                    if status_code == -1
                    else f"Connection failed with status code: {status_code}"
                )
            )
        }

    @app.post("/llm/config")
    def llm_config_api(req: LLMConfigRequest):
        settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            status_code = apply_llm_conf(
                req.api_key, req.api_base, req.language_model, req.max_tokens, origin_call="http"
            )
        elif req.llm_type == "qianfan_wenxin":
            status_code = apply_llm_conf(req.api_key, req.secret_key, req.language_model, None, origin_call="http")
        else:
            status_code = apply_llm_conf(req.host, req.port, req.language_model, None, origin_call="http")
        return {
            "message": (
                "Connection successful. Configured finished."
                if 200 <= status_code < 300
                else (
                    "Unsupported HTTP method"
                    if status_code == -1
                    else f"Connection failed with status code: {status_code}"
                )
            )
        }

    @app.post("/embedding/config")
    def embedding_config_api(req: LLMConfigRequest):
        settings.embedding_type = req.llm_type

        if req.llm_type == "openai":
            status_code = apply_embedding_conf(req.api_key, req.api_base, req.language_model, origin_call="http")
        elif req.llm_type == "qianfan_wenxin":
            status_code = apply_embedding_conf(req.api_key, req.api_base, None, origin_call="http")
        else:
            status_code = apply_embedding_conf(req.host, req.port, req.language_model, origin_call="http")
        return {
            "message": (
                "Connection successful. Configured finished."
                if 200 <= status_code < 300
                else (
                    "Unsupported HTTP method"
                    if status_code == -1
                    else f"Connection failed with status code: {status_code}"
                )
            )
        }
