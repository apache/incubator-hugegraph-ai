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
from typing import AsyncGenerator, Generator, Optional, List, Dict, Any, Callable

from openai import OpenAI, AsyncOpenAI
from retry import retry

from hugegraph_llm.config import llm_settings
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log


class QianfanClient(BaseLLM):
    def __init__(self, model_name: Optional[str] = "ernie-3.5-8k",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or llm_settings.qianfan_chat_api_key,
            base_url=base_url or llm_settings.qianfan_base_url,
        )
        self.aclient = AsyncOpenAI(
            api_key=api_key or llm_settings.qianfan_chat_api_key,
            base_url=base_url or llm_settings.qianfan_base_url,
        )
        self.chat_model = model_name

    @retry(tries=3, delay=1)
    def generate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages
        )

        log.info("Token usage: %s", json.dumps({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }))
        return response.choices[0].message.content

    @retry(tries=3, delay=1)
    async def agenerate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        response = await self.aclient.chat.completions.create(
            model=self.chat_model,
            messages=messages
        )

        log.info("Token usage: %s", json.dumps({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }))
        return response.choices[0].message.content

    def generate_streaming(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
            on_token_callback: Optional[Callable] = None,
    ) -> Generator[str, None, None]:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                if on_token_callback:
                    on_token_callback(token)
                yield token

    async def agenerate_streaming(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
            on_token_callback: Optional[Callable] = None,
    ) -> AsyncGenerator[str, None]:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        try:
            stream = await self.aclient.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    if on_token_callback:
                        on_token_callback(token)
                    yield token
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise e

    def num_tokens_from_string(self, string: str) -> int:
        return len(string)

    def max_allowed_token_length(self) -> int:
        return 6000

    def get_llm_type(self) -> str:
        return "qianfan_wenxin_v2"


if __name__ == "__main__":
    client = QianfanClient()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))

