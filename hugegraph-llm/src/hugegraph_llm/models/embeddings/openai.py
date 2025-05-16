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


class OpenAIEmbedding:
    def __init__(
            self,
            model_name: str = "text-embedding-3-small",
            api_key: Optional[str] = None,
            api_base: Optional[str] = None
    ):
        api_key = api_key or ''
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.embedding_model_name = model_name

    def get_text_embedding(self, text: str) -> List[float]:
        """Comment"""
        response = self.client.embeddings.create(input=text, model=self.embedding_model_name)
        return response.data[0].embedding

    def get_texts_embeddings(
            self,
            texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for multiple texts in a single batch.
        
        This method efficiently processes multiple texts at once by leveraging
        OpenAI's batching capabilities, which is more efficient than processing
        texts individually.
        
        Parameters
        ----------
        texts : List[str]
            A list of text strings to be embedded.
            
        Returns
        -------
        List[List[float]]
            A list of embedding vectors, where each vector is a list of floats.
            The order of embeddings matches the order of input texts.
        """
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        return [data.embedding for data in response.data]

    async def async_get_text_embedding(self, text: str) -> List[float]:
        """Comment"""
        response = await self.aclient.embeddings.create(input=text, model=self.embedding_model_name)
        return response.data[0].embedding
