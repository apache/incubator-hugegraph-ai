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

from tqdm import tqdm
from hugegraph_llm.config import settings, resource_path
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.log import log


class BuildVectorIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.embedding = embedding
        self.index_dir = str(os.path.join(resource_path, settings.graph_name, "chunks"))
        self.vector_index = VectorIndex.from_index_file(self.index_dir)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if "chunks" not in context:
            raise ValueError("chunks not found in context.")
        chunks = context["chunks"]
        chunks_embedding = []
        log.debug("Building vector index for %s chunks...", len(context["chunks"]))
        for chunk in tqdm(chunks):
            chunks_embedding.append(self.embedding.get_text_embedding(chunk))
        if len(chunks_embedding) > 0:
            self.vector_index.add(chunks_embedding, chunks)
            self.vector_index.to_index_file(self.index_dir)
        return context
