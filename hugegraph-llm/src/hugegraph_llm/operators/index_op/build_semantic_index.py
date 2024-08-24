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

from tqdm import tqdm
from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.utils.log import log


class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.content_file = str(os.path.join(resource_path, settings.graph_name, "vid.pkl"))
        self.index_file = str(os.path.join(resource_path, settings.graph_name, "vid.faiss"))
        self.embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if len(context["vertices"]) > 0:
            log.debug("Building vector index for %s vertices...", len(context["vertices"]))
            vids = []
            vids_embedding = []
            for vertex in tqdm(context["vertices"]):
                vertex_text = f"{vertex['label']}\n{vertex['properties']}"
                vids_embedding.append(self.embedding.get_text_embedding(vertex_text))
                vids.append(vertex["id"])
            vids_embedding = [self.embedding.get_text_embedding(vid) for vid in vids]
            log.debug("Vector index built for %s vertices.", len(vids))
            if os.path.exists(self.index_file) and os.path.exists(self.content_file):
                vector_index = VectorIndex.from_index_file(self.index_file, self.content_file)
            else:
                vector_index = VectorIndex(len(vids_embedding[0]))
            vector_index.add(vids_embedding, vids)
            vector_index.to_index_file(self.index_file, self.content_file)
        else:
            log.debug("No vertices to build vector index.")
        return context
