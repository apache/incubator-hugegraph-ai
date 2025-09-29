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


import asyncio
import os
from typing import Any, Dict

from hugegraph_llm.config import resource_path, huge_settings, llm_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.embedding_utils import (
    get_embeddings_parallel,
    get_filename_prefix,
    get_index_folder_name,
)
from hugegraph_llm.utils.log import log


class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.folder_name = get_index_folder_name(
            huge_settings.graph_name, huge_settings.graph_space
        )
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "graph_vids"))
        self.filename_prefix = get_filename_prefix(
            llm_settings.embedding_type, getattr(embedding, "model_name", None)
        )
        self.vid_index = VectorIndex.from_index_file(self.index_dir, self.filename_prefix)
        self.embedding = embedding
        self.sm = SchemaManager(huge_settings.graph_name)

    def _extract_names(self, vertices: list[str]) -> list[str]:
        return [v.split(":")[1] for v in vertices]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        all_pk_flag = all(data.get("id_strategy") == "PRIMARY_KEY" for data in vertexlabels)

        past_vids = self.vid_index.properties
        # TODO: We should build vid vector index separately, especially when the vertices may be very large

        present_vids = context["vertices"]  # Warning: data truncated by fetch_graph_data.py
        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids) - set(past_vids))

        if added_vids:
            vids_to_process = self._extract_names(added_vids) if all_pk_flag else added_vids
            added_embeddings = asyncio.run(get_embeddings_parallel(self.embedding, vids_to_process))
            log.info("Building vector index for %s vertices...", len(added_vids))
            self.vid_index.add(added_embeddings, added_vids)
            self.vid_index.to_index_file(self.index_dir, self.filename_prefix)
        else:
            log.debug("No update vertices to build vector index.")
        context.update(
            {
                "removed_vid_vector_num": removed_num,
                "added_vid_vector_num": len(added_vids),
            }
        )
        return context
