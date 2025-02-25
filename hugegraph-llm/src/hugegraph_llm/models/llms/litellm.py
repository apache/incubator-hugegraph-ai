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

from typing import Callable, List, Optional, Dict, Any

import tiktoken
from litellm import completion, acompletion
from litellm.exceptions import RateLimitError, BudgetExceededError, APIError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log


class LiteLLMClient(BaseLLM):
    """Wrapper for LiteLLM Client that supports multiple LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4",  # Can be any model supported by LiteLLM
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, BudgetExceededError, APIError))
    )
    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a response to the query messages/prompt."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        try:
            print("base_url:" + self.api_base)
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.api_base,
            )
            log.info("Token usage: %s", response.usage)
            return response.choices[0].message.content
        except Exception as e:
            log.error("Error in LiteLLM call: %s", e)
            return f"Error: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, BudgetExceededError, APIError))
    )
    async def agenerate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        """Generate a response to the query messages/prompt asynchronously."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.api_base,
            )
            log.info("Token usage: %s", response.usage)
            return response.choices[0].message.content
        except Exception as e:
            log.error("Error in async LiteLLM call: %s", e)
            return f"Error: {str(e)}"

    def generate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Callable = None,
    ) -> str:
        """Generate a response to the query messages/prompt in streaming mode."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.api_base,
                stream=True,
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                if on_token_callback:
                    on_token_callback(chunk)
            return result
        except Exception as e:
            log.error("Error in streaming LiteLLM call: %s", e)
            return f"Error: {str(e)}"

    def num_tokens_from_string(self, string: str) -> int:
        """Get token count from string."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except Exception:
            # Fallback for models not supported by tiktoken
            # Rough estimate: 1 token ≈ 4 characters
            return len(string) // 4

    def max_allowed_token_length(self) -> int:
        """Get max-allowed token length based on the model."""
        return 4096  # Default to 4096 if model not found

    def get_llm_type(self) -> str:
        return "litellm" 