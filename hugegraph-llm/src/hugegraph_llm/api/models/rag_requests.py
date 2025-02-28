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
    max_graph_items: int = Query(30, description="Maximum number of items for GQL queries in graph.")
    topk_return_results: int = Query(20, description="Number of sorted results to return finally.")
    vector_dis_threshold: float = Query(0.9, description="Threshold for vector similarity\
                                         (results greater than this will be ignored).")
    topk_per_keyword : int = Query(1, description="TopK results returned for each keyword \
                                   extracted from the query, by default only the most similar one is returned.")

    ip: str = Query('127.0.0.1', description="graph server ip.")
    port: str = Query('8080', description="graph server port.")
    name: str = Query('hugegraph', description="graph name.")
    user: str = Query('xxx', description="graph server username.")
    pwd: str = Query('xxx', description="graph server pwd.")
    gs: str = Query('', description="graphspace.")


# TODO: import the default value of prompt.* dynamically
class GraphRAGRequest(BaseModel):
    query: str = Query("", description="Query you want to ask")
    gremlin_tmpl_num: int = Query(
        1, description="Number of Gremlin templates to use. If num <=0 means template is not provided"
    )
    rerank_method: Literal["bleu", "reranker"] = Query("bleu", description="Method to rerank the results.")
    near_neighbor_first: bool = Query(False, description="Prioritize near neighbors in the search results.")
    custom_priority_info: str = Query("", description="Custom information to prioritize certain results.")
    gremlin_prompt: Optional[str] = Query(
        prompt.gremlin_generate_prompt,
        description="Prompt for the Text2Gremlin query.",
    )
    max_graph_items: int = Query(30, description="Maximum number of items for GQL queries in graph.")
    topk_return_results: int = Query(20, description="Number of sorted results to return finally.")
    vector_dis_threshold: float = Query(0.9, description="Threshold for vector similarity \
                                        (results greater than this will be ignored).")
    topk_per_keyword : int = Query(1, description="TopK results returned for each keyword extracted\
                                    from the query, by default only the most similar one is returned.")

    ip: str = Query('127.0.0.1', description="graph server ip.")
    port: str = Query('8080', description="graph server port.")
    name: str = Query('hugegraph', description="graph name.")
    user: str = Query('xxx', description="graph server username.")
    pwd: str = Query('xxx', description="graph server pwd.")
    gs: str = Query('', description="graphspace.")


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
