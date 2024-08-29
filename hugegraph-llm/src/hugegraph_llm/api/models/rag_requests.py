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

from typing import Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    raw_llm: Optional[bool] = True
    vector_only: Optional[bool] = False
    graph_only: Optional[bool] = False
    graph_vector: Optional[bool] = False
    answer_prompt: Optional[str] = None


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
