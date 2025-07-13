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


from typing import Optional, List

from openai import OpenAI, AsyncOpenAI

from hugegraph_llm.config import llm_settings

"""
"QianFan" platform can be understood as a unified LLM platform that encompasses the 
WenXin large model along with other 
common open-source models. 

It enables the invocation and switching between WenXin and these open-source models.
"""


class QianFanEmbedding:
    def __init__(
            self,
            model_name: str = "embedding-v1",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None
    ):
        self.client = OpenAI(
            api_key=api_key or llm_settings.qianfan_embedding_api_key,
            base_url=base_url or llm_settings.qianfan_base_url,
        )
        self.aclient = AsyncOpenAI(
            api_key=api_key or llm_settings.qianfan_embedding_api_key,
            base_url=base_url or llm_settings.qianfan_base_url,
        )
        self.embedding_model_name = model_name

    def get_text_embedding(self, text: str) -> List[float]:
        """ Usage refer v2 API documentation"""
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=[text]
        )
        return response.data[0].embedding

    def get_texts_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ Usage refer v2 API documentation"""
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=texts
        )
        return [data.embedding for data in response.data]

    async def async_get_text_embedding(self, text: str) -> List[float]:
        """ Usage refer v2 API documentation"""
        response = await self.aclient.embeddings.create(
            model=self.embedding_model_name,
            input=[text]
        )
        return response.data[0].embedding