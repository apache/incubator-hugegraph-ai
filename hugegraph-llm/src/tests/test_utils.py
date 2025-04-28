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

import os
from unittest.mock import MagicMock, patch


# 检查是否应该跳过外部服务测试
def should_skip_external():
    return os.environ.get("SKIP_EXTERNAL_SERVICES") == "true"


# 创建模拟的 Ollama 嵌入响应
def mock_ollama_embedding(dimension=1024):
    return {"embedding": [0.1] * dimension}


# 创建模拟的 OpenAI 嵌入响应
def mock_openai_embedding(dimension=1536):
    class MockResponse:
        def __init__(self, data):
            self.data = data

    return MockResponse([{"embedding": [0.1] * dimension, "index": 0}])


# 创建模拟的 OpenAI 聊天响应
def mock_openai_chat_response(text="模拟的 OpenAI 响应"):
    class MockResponse:
        def __init__(self, content):
            self.choices = [MagicMock()]
            self.choices[0].message.content = content

    return MockResponse(text)


# 创建模拟的 Ollama 聊天响应
def mock_ollama_chat_response(text="模拟的 Ollama 响应"):
    return {"message": {"content": text}}


# 装饰器，用于模拟 Ollama 嵌入
def with_mock_ollama_embedding(func):
    @patch("ollama._client.Client._request_raw")
    def wrapper(self, mock_request, *args, **kwargs):
        mock_request.return_value.json.return_value = mock_ollama_embedding()
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器，用于模拟 OpenAI 嵌入
def with_mock_openai_embedding(func):
    @patch("openai.resources.embeddings.Embeddings.create")
    def wrapper(self, mock_create, *args, **kwargs):
        mock_create.return_value = mock_openai_embedding()
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器，用于模拟 Ollama LLM 客户端
def with_mock_ollama_client(func):
    @patch("ollama._client.Client._request_raw")
    def wrapper(self, mock_request, *args, **kwargs):
        mock_request.return_value.json.return_value = mock_ollama_chat_response()
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器，用于模拟 OpenAI LLM 客户端
def with_mock_openai_client(func):
    @patch("openai.resources.chat.completions.Completions.create")
    def wrapper(self, mock_create, *args, **kwargs):
        mock_create.return_value = mock_openai_chat_response()
        return func(self, *args, **kwargs)

    return wrapper


# 下载 NLTK 资源的辅助函数
def ensure_nltk_resources():
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# 创建测试文档的辅助函数
def create_test_document(content="这是一个测试文档"):
    from hugegraph_llm.document.document import Document

    return Document(content=content, metadata={"source": "test"})


# 创建测试向量索引的辅助函数
def create_test_vector_index(dimension=1536):
    from hugegraph_llm.indices.vector_index import VectorIndex

    index = VectorIndex(dimension)
    return index
