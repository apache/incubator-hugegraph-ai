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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from litellm import APIConnectionError, APIError, RateLimitError

from hugegraph_llm.models.embeddings.litellm import LiteLLMEmbedding

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError(message="temporary 429", llm_provider="hosted_vllm", model="test"),
        APIConnectionError(message="temporary connection failure", llm_provider="hosted_vllm", model="test"),
        APIError(status_code=500, message="temporary API failure", llm_provider="hosted_vllm", model="test"),
    ],
    ids=["rate_limit", "connection", "api_error"],
)
def test_batch_retry_preserves_completed_batches_and_order(error):
    embedding = LiteLLMEmbedding(embedding_dimension=1, api_key="test", model_name="hosted_vllm/test")
    batches = []

    def respond(**kwargs):
        texts = kwargs["input"]
        batches.append(texts)
        if len(batches) == 2:
            raise error
        return SimpleNamespace(data=[{"embedding": [float(text)]} for text in texts], usage=None)

    with (
        patch("hugegraph_llm.models.embeddings.litellm.embedding", side_effect=respond),
        patch("tenacity.nap.time.sleep"),
    ):
        result = embedding.get_texts_embeddings(["0", "1", "2", "3", "4"], batch_size=2)

    assert batches == [["0", "1"], ["2", "3"], ["2", "3"], ["4"]]
    assert result == [[float(i)] for i in range(5)]


def test_non_retryable_batch_error_is_not_retried():
    embedding = LiteLLMEmbedding(embedding_dimension=1, api_key="test")
    error = ValueError("invalid embedding input")

    with patch("hugegraph_llm.models.embeddings.litellm.embedding", side_effect=error) as mock_embedding:
        with pytest.raises(ValueError) as raised:
            embedding.get_texts_embeddings(["test"])

    assert raised.value is error
    mock_embedding.assert_called_once()
