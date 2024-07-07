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
from typing import Any, Dict

from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.utils.log import log


class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.content_file = os.path.join(resource_path, settings.graph_name, "vid.pkl")
        self.index_file = os.path.join(resource_path, settings.graph_name, "vid.faiss")
        self.embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        vids = [vertex["id"] for vertex in context["vertices"]]
        if len(vids) > 0:
            log.debug("Building vector index for %s vertices...", len(vids))
            vids_embedding = [self.embedding.get_text_embedding(vid) for vid in vids]
            log.debug("Vector index built for %s vertices.", len(vids))
            vector_index = VectorIndex(len(vids_embedding[0]))
            vector_index.add(vids_embedding, vids)
            vector_index.to_index_file(str(self.index_file), str(self.content_file))
        else:
            log.debug("No vertices to build vector index.")
        return context
