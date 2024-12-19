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

from typing import Optional, Literal

from fastapi import Query
from pydantic import BaseModel

from hugegraph_llm.config import prompt


class RAGRequest(BaseModel):
    query: str = Query("", description="Query you want to ask")
    raw_answer: bool = Query(False, description="Use LLM to generate answer directly")
    vector_only: bool = Query(False, description="Use LLM to generate answer with vector")
    graph_only: bool = Query(True, description="Use LLM to generate answer with graph RAG only")
    graph_vector_answer: bool = Query(False, description="Use LLM to generate answer with vector & GraphRAG")
    graph_ratio: float = Query(0.5, description="The ratio of GraphRAG ans & vector ans")
    rerank_method: Literal["bleu", "reranker"] = Query("bleu", description="Method to rerank the results.")
    near_neighbor_first: bool = Query(False, description="Prioritize near neighbors in the search results.")
    with_gremlin_tmpl: bool = Query(True, description="Use example template in text2gremlin")
    custom_priority_info: str = Query("", description="Custom information to prioritize certain results.")
    answer_prompt: Optional[str] = Query(prompt.answer_prompt, description="Prompt to guide the answer generation.")
    keywords_extract_prompt: Optional[str] = Query(
        prompt.keywords_extract_prompt,
        description="Prompt for extracting keywords from query.",
    )
    gremlin_tmpl_num: int = Query(1, description="Number of Gremlin templates to use.")
    gremlin_prompt: Optional[str] = Query(
        prompt.gremlin_generate_prompt,
        description="Prompt for the Text2Gremlin query.",
    )


class GraphRAGRequest(BaseModel):
    query: str = Query("", description="Query you want to ask")
    gremlin_tmpl_num: int = Query(1, description="Number of Gremlin templates to use.")
    with_gremlin_tmpl: bool = Query(True, description="Use example template in text2gremlin")
    answer_prompt: Optional[str] = Query(prompt.answer_prompt, description="Prompt to guide the answer generation.")
    rerank_method: Literal["bleu", "reranker"] = Query("bleu", description="Method to rerank the results.")
    near_neighbor_first: bool = Query(False, description="Prioritize near neighbors in the search results.")
    custom_priority_info: str = Query("", description="Custom information to prioritize certain results.")
    gremlin_prompt: Optional[str] = Query(
        prompt.gremlin_generate_prompt,
        description="Prompt for the Text2Gremlin query.",
    )


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


class RerankerConfigRequest(BaseModel):
    reranker_model: str
    reranker_type: str
    api_key: str
    cohere_base_url: Optional[str] = None


class LogStreamRequest(BaseModel):
    admin_token: Optional[str] = None
    log_file: Optional[str] = "llm-server.log"
