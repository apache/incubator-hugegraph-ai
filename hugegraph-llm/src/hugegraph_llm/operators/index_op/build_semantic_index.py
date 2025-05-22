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

    async def _get_embeddings_parallel(self, vids: list[str]) -> list[Any]:
        sem = asyncio.Semaphore(10)
        batch_size = 1000
        async def get_embeddings_with_semaphore(vid_list: list[str]) -> Any:
            # Executes sync embedding method in a thread pool via loop.run_in_executor, combining async programming
            # with multi-threading capabilities.
            # This pattern avoids blocking the event loop and prepares for a future fully async pipeline.
            async with sem:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embedding.get_texts_embeddings, vid_list)

        # Split vids into batches of size batch_size
        vid_batches = [vids[i:i + batch_size] for i in range(0, len(vids), batch_size)]

        # Create tasks for each batch
        tasks = [get_embeddings_with_semaphore(batch) for batch in vid_batches]

        embeddings = []
        with tqdm(total=len(tasks)) as pbar:
            for future in asyncio.as_completed(tasks):
                batch_embeddings = await future
                embeddings.extend(batch_embeddings) # Extend the list with batch results
                pbar.update(1)
        return embeddings

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        all_pk_flag = all(data.get('id_strategy') == 'PRIMARY_KEY' for data in vertexlabels)

        past_vids = self.vid_index.properties
        # TODO: We should build vid vector index separately, especially when the vertices may be very large
        present_vids = context["vertices"] # Warning: data truncated by fetch_graph_data.py
        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids) - set(past_vids))

        if added_vids:
            vids_to_process = self._extract_names(added_vids) if all_pk_flag else added_vids
            added_embeddings = asyncio.run(self._get_embeddings_parallel(vids_to_process))
            log.info("Building vector index for %s vertices...", len(added_vids))
            self.vid_index.add(added_embeddings, added_vids)
            self.vid_index.to_index_file(self.index_dir)
        else:
            log.debug("No update vertices to build vector index.")
        context.update({
            "removed_vid_vector_num": removed_num,
            "added_vid_vector_num": len(added_vids)
        })
        return context
