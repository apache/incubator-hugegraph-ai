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



from hugegraph_llm.models.embeddings.openai import OpenAIEmbedding
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
from hugegraph_llm.config import settings


class Embeddings:
    def __init__(self):
        self.embedding_type = settings.embedding_type

    def get_embedding(self):
        if self.embedding_type == "openai":
            return OpenAIEmbedding(
                model_name=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
            )
        if self.embedding_type == "ollama":
            return OllamaEmbedding(
                model=settings.ollama_embedding_model,
                host=settings.ollama_host,
                port=settings.ollama_port
            )
        raise Exception("embedding type is not supported !")
