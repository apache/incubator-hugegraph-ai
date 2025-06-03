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


from typing import List

import ollama

from .base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model: str, host: str = "127.0.0.1", port: int = 11434, **kwargs):
        self.model_name = model
        self.client = ollama.Client(host=f"http://{host}:{port}", **kwargs)
        self.async_client = ollama.AsyncClient(host=f"http://{host}:{port}", **kwargs)
        self.embedding_dimension = None

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.get_texts_embeddings([text])[0]

    def get_texts_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a single batch.

        Returns
        -------
        List[List[float]]
            A list of embedding vectors, where each vector is a list of floats.
            The order of embeddings matches the order of input texts.
        """
        if not hasattr(self.client, "embed"):
            error_message = (
                "The required 'embed' method was not found on the Ollama client. "
                "Please ensure your ollama library is up-to-date and supports batch embedding. "
            )
            raise AttributeError(error_message)

        response = self.client.embed(model=self.model_name, input=texts)["embeddings"]
        return [list(inner_sequence) for inner_sequence in response]

    # TODO: Add & implement batch processing for async_get_texts_embeddings (refactor here)
    async def async_get_text_embedding(self, text: str) -> List[float]:
        response = await self.async_client.embeddings(model=self.model_name, prompt=text)
        return list(response["embedding"])
