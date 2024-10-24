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
        self.index_dir = str(os.path.join(resource_path, settings.graph_name, "graph_vids"))
        self.vid_index = VectorIndex.from_index_file(self.index_dir)
        self.embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        past_vids = self.vid_index.properties
        present_vids = context["vertices"]
        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids) - set(past_vids))
        if len(added_vids) > 0:
            log.debug("Building vector index for %s vertices...", len(added_vids))
            added_embeddings = [self.embedding.get_text_embedding(v) for v in tqdm(added_vids)]
            log.debug("Vector index built for %s vertices.", len(added_embeddings))
            self.vid_index.add(added_embeddings, added_vids)
            self.vid_index.to_index_file(self.index_dir)
        else:
            log.debug("No vertices to build vector index.")
        context.update({
            "removed_vid_vector_num": removed_num,
            "added_vid_vector_num": len(added_vids)
        })
        return context
