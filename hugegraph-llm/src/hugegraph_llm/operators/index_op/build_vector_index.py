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
from typing import Dict, Any

from hugegraph_llm.config import settings, resource_path
from hugegraph_llm.indices.vector_index import VectorIndex


class BuildVectorIndex:
    def __init__(self, context_key: str = "chunks", embedding_key: str = "chunks_embedding"):
        self.index_file = os.path.join(resource_path, settings.graph_name, "index.faiss")
        self.content_file = os.path.join(resource_path, settings.graph_name, "properties.pkl")
        if not os.path.exists(os.path.join(resource_path, settings.graph_name)):
            os.mkdir(os.path.join(resource_path, settings.graph_name))
        self.context_key = context_key
        self.embedding_key = embedding_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context[self.context_key]
        chunks_embedding = context[self.embedding_key]
        if len(chunks_embedding) > 0:
            vector_index = VectorIndex(len(chunks_embedding[0]))
            vector_index.add(chunks_embedding, chunks)
            vector_index.to_index_file(str(self.index_file), str(self.content_file))
        return context
