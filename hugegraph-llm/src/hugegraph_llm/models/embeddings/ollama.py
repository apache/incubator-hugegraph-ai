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
    def __init__(
            self,
            model: str,
            host: str = "127.0.0.1",
            port: int = 11434,
            **kwargs
    ):
        self.model = model
        self.client = ollama.Client(host=f"http://{host}:{port}", **kwargs)
        self.async_client = ollama.AsyncClient(host=f"http://{host}:{port}", **kwargs)
        self.embedding_dimension = None

    def get_text_embedding(
            self,
            text: str
    ) -> List[float]:
        """Get embedding for a single text.

        This method handles different Ollama client API versions by checking for
        the presence of 'embed' or 'embeddings' methods.
        """
        if hasattr(self.client, "embed"):
            response = self.client.embed(model=self.model, input=text)
            try:
                # First, try the structure typically seen for single embeddings
                # or newer batch responses that might return a single "embedding" key.
                return list(response["embedding"])
            except KeyError:
                # Fallback for older batch-like response for single item,
                # or if "embeddings" is a list with one item.
                try:
                    return list(response["embeddings"][0])
                except (KeyError, IndexError) as e:
                    raise RuntimeError(
                        "Failed to extract embedding from Ollama client 'embed' response. "
                        f"Response: {response}. Error: {e}"
                    )
        elif hasattr(self.client, "embeddings"):
            response = self.client.embeddings(model=self.model, prompt=text)
            try:
                return list(response["embedding"])
            except KeyError as e:
                raise RuntimeError(
                    "Failed to extract embedding from Ollama client 'embeddings' response. "
                    f"Response: {response}. Error: {e}"
                )
        else:
            raise AttributeError(
                "Ollama client object has neither 'embed' nor 'embeddings' method. "
                "Please check your ollama library version."
            )

    def get_texts_embeddings(
            self,
            texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for multiple texts in a single batch.
        
        This method efficiently processes multiple texts at once by leveraging
        Ollama's batching capabilities, which is more efficient than processing
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
        if hasattr(self.client, "embed"):
            response = self.client.embed(model=self.model, input=texts)["embeddings"]
            return [list(inner_sequence) for inner_sequence in response]
        elif hasattr(self.client, "embeddings"):
            embeddings_list = []
            for text_item in texts:
                response_item = self.client.embeddings(model=self.model, prompt=text_item)
                embeddings_list.append(list(response_item["embedding"]))
            return embeddings_list
        else:
            raise AttributeError(
                "Ollama client object has neither 'embed' nor 'embeddings' method. "
                "Please check your ollama library version."
            )

    async def async_get_text_embedding(
            self,
            text: str
    ) -> List[float]:
        """Comment"""
        response = await self.async_client.embeddings(model=self.model, prompt=text)
        return list(response["embedding"])
