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

from hugegraph_llm.config import resource_path, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.log import log
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager

class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.index_dir = str(os.path.join(resource_path, huge_settings.graph_name, "graph_vids"))
        self.vid_index = VectorIndex.from_index_file(self.index_dir)
        self.embedding = embedding
        self.sm = SchemaManager(huge_settings.graph_name)

    def _extract_names(self, vertices: list[str]) -> list[str]:
        return [v.split(":")[1] for v in vertices]

    def _check_primary_key(self, vertexlabels):
        return all(data.get('id_strategy') == 'PRIMARY_KEY' for data in vertexlabels)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        flag_extract_names = self._check_primary_key(self.sm.schema.getSchema()["vertexlabels"])
        past_vids = self.vid_index.properties
        # TODO: We should build vid vector index separately, especially when the vertices may be very large
        present_vids = context["vertices"] # Warning: data truncated by fetch_graph_data.py
        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids) - set(past_vids))

        if len(added_vids) > 0:
            # TODO: We should use multi value map when meet same value. (e.g [1:tom, 2:tom, tom] in one graph)
            if flag_extract_names:
                extract_added_vids = self._extract_names(added_vids)
                added_embeddings = [self.embedding.get_text_embedding(v) for v in tqdm(extract_added_vids)]
            else:
                added_embeddings = [self.embedding.get_text_embedding(v) for v in tqdm(added_vids)]
            log.info("Building vector index for %s vertices...", len(added_vids))
            log.info("Vector index built for %s vertices.", len(added_embeddings))
            self.vid_index.add(added_embeddings, added_vids)
            self.vid_index.to_index_file(self.index_dir)
        else:
            log.debug("No update vertices to build vector index.")
        context.update({
            "removed_vid_vector_num": removed_num,
            "added_vid_vector_num": len(added_vids)
        })
        return context
