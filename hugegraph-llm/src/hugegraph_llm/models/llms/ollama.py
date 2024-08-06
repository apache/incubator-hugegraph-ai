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


from typing import Any, List, Optional, Callable, Dict

import ollama
from retry import retry

from .base import BaseLLM


class OllamaClient(BaseLLM):
    """LLM wrapper should take in a prompt and return a string."""
    def __init__(self, model: str, host: str = "127.0.0.1", port: int = 11434, **kwargs):
        self.model = model
        self.client = ollama.Client(host=f"http://{host}:{port}", **kwargs)
        self.async_client = ollama.AsyncClient(host=f"http://{host}:{port}", **kwargs)

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Comment"""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise e

    @retry(tries=3, delay=1)
    async def agenerate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        """Comment"""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.async_client.chat(
                model=self.model,
                messages=messages,
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise e

    def generate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Callable = None,
    ) -> List[Any]:
        """Comment"""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        chunks = []
        for chunk in stream:
            on_token_callback(chunk["message"]["content"])
            chunks.append(chunk)
        return chunks

    def num_tokens_from_string(
        self,
        string: str,
    ) -> int:
        """Given a string returns the number of tokens the given string consists of"""
        return len(string)

    def max_allowed_token_length(
        self,
    ) -> int:
        """Returns the maximum number of tokens the LLM can handle"""
        return 4096

    def get_llm_type(self) -> str:
        """Returns the type of the LLM"""
        return "ollama"
