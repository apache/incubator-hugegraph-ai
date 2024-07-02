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


from typing import Any, Dict

from hugegraph_llm.models.embeddings.base import BaseEmbedding


class ChunkEmbedding:
    def __init__(
            self,
            embedding: BaseEmbedding,
            context_key: str = "chunks",
            result_key: str = "chunks_embedding"
    ):
        self.embedding = embedding
        self.context_key = context_key
        self.result_key = result_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context[self.context_key]
        chunks_embedding = []
        for chunk in chunks:
            chunks_embedding.append(self.embedding.get_text_embedding(str(chunk)))
        context[self.result_key] = chunks_embedding
        return context
