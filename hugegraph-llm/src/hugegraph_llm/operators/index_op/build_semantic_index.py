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
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from hugegraph_llm.config import resource_path, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient

INDEX_PROPERTY_GREMLIN = """
g.V().hasLabel('{label}')
 .limit(100000)
 .project('vid', 'properties')
 .by(id())
 .by(valueMap({fields}))
 .toList()
"""

class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding):
        self.index_dir = str(os.path.join(resource_path, huge_settings.graph_name, "graph_vids"))
        self.index_dir_prop = str(os.path.join(resource_path, huge_settings.graph_name, "graph_props"))
        self.vid_index = VectorIndex.from_index_file(self.index_dir)
        self.prop_index = VectorIndex.from_index_file(self.index_dir_prop)
        self.embedding = embedding
        self.sm = SchemaManager(huge_settings.graph_name)
        self.client = PyHugeClient(
            url=huge_settings.graph_url,
            graph=huge_settings.graph_name,
            user=huge_settings.graph_user,
            pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )

    def _extract_names(self, vertices: list[str]) -> list[str]:
        return [v.split(":")[1] for v in vertices]

    async def _get_embeddings_parallel(self, vids: list[str]) -> list[Any]:
        sem = asyncio.Semaphore(10)
        batch_size = 1000

        # TODO: refactor the logic here (call async method)
        async def get_embeddings_with_semaphore(vid_list: list[str]) -> Any:
            # Executes sync embedding method in a thread pool via loop.run_in_executor, combining async programming
            # with multi-threading capabilities.
            # This pattern avoids blocking the event loop and prepares for a future fully async pipeline.
            async with sem:
                loop = asyncio.get_running_loop()
                # FIXME: [PR-238] add & use async_get_texts_embedding instead of sync method
                return await loop.run_in_executor(None, self.embedding.get_texts_embeddings, vid_list)

        # Split vids into batches of size batch_size
        vid_batches = [vids[i:i + batch_size] for i in range(0, len(vids), batch_size)]

        # Create tasks for each batch
        tasks = [get_embeddings_with_semaphore(batch) for batch in vid_batches]

        embeddings = []
        with tqdm(total=len(tasks)) as pbar:
            for future in asyncio.as_completed(tasks):
                batch_embeddings = await future
                embeddings.extend(batch_embeddings)  # Extend the list with batch results
                pbar.update(1)
        return embeddings

    def diff_property_sets(
        self,
        present_prop_value_to_propset: dict,
        past_prop_value_to_propset: dict
    ):
        to_add = []
        to_update = []
        to_update_remove = []
        to_remove_keys = set(past_prop_value_to_propset) - set(present_prop_value_to_propset)
        to_remove = [past_prop_value_to_propset[k] for k in to_remove_keys]
        for prop_value, present_propset in present_prop_value_to_propset.items():
            past_propset = past_prop_value_to_propset.get(prop_value)
            if past_propset is None:
                to_add.append((prop_value, present_propset))
            elif present_propset != past_propset:
                to_update.append((prop_value, present_propset))
                to_update_remove.append((prop_value, past_propset))
        return to_add, to_update, to_remove, to_update_remove

    def get_present_props(self, context: dict[str, Any]) -> dict[str, frozenset[tuple[str, str]]]:
        results = []
        for item in context["index_labels"]:
            label = item["base_value"]
            fields = item["fields"]
            fields_str_list = [f"'{field}'" for field in fields]
            fields_for_query = ", ".join(fields_str_list)
            gremlin_query = INDEX_PROPERTY_GREMLIN.format(
                label=label,
                fields=fields_for_query
            )
            log.debug("gremlin_query: %s", gremlin_query)
            result = self.client.gremlin().exec(gremlin=gremlin_query)["data"]
            results.extend(result)
        orig_present_prop_value_to_propset = defaultdict(set)
        for item in results:
            properties = item["properties"]
            for prop_key, values in properties.items():
                if not values:
                    continue
                prop_value = str(values[0])
                orig_present_prop_value_to_propset[prop_value].add((prop_key, prop_value))
        present_prop_value_to_propset = {
            k: frozenset(v)
            for k, v in orig_present_prop_value_to_propset.items()
        }
        return present_prop_value_to_propset

    def get_past_props(self) -> dict[str, frozenset[tuple[str, str]]]:
        orig_past_prop_value_to_propset = defaultdict(set)
        for propset in self.prop_index.properties:
            for _, prop_value in propset:
                orig_past_prop_value_to_propset[prop_value].update(propset)
        past_prop_value_to_propset = {
            k: frozenset(v)
            for k, v in orig_past_prop_value_to_propset.items()
        }
        return past_prop_value_to_propset

    def run(self, context: dict[str, Any]) -> dict[str, Any]:  # pylint: disable=too-many-statements, too-many-branches
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        all_pk_flag = all(data.get('id_strategy') == 'PRIMARY_KEY' for data in vertexlabels)

        past_vids = self.vid_index.properties
        # TODO: We should build vid vector index separately, especially when the vertices may be very large
        present_vids = context["vertices"]  # Warning: data truncated by fetch_graph_data.py
        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        if removed_num:
            self.vid_index.to_index_file(self.index_dir)
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

        if context["index_labels"]:
            present_prop_value_to_propset = self.get_present_props(context)
            # log.debug("present_prop_value_to_propset: %s", present_prop_value_to_propset)
            past_prop_value_to_propset = self.get_past_props()
            # log.debug("past_prop_value_to_propset: %s", past_prop_value_to_propset)
            to_add, to_update, to_remove, to_update_remove = self.diff_property_sets(
                present_prop_value_to_propset,
                past_prop_value_to_propset
            )
            log.debug("to_add: %s", to_add)
            log.debug("to_update: %s", to_update)
            log.debug("to_remove: %s", to_remove)
            log.debug("to_update_remove: %s", to_update_remove)
            log.info("Removing %s outdated property value", len(to_remove))
            removed_props_num = self.prop_index.remove(to_remove)
            if removed_props_num:
                self.prop_index.to_index_file(self.index_dir_prop)
            all_to_add = to_add + to_update
            add_propsets = []
            add_prop_values = []
            for prop_value, propset in all_to_add:
                add_propsets.append(propset)
                add_prop_values.append(prop_value)
            if add_prop_values:
                if len(add_prop_values) > 100000:
                    log.warning("The number of props > 100000, please select which properties to vectorize.")
                    context.update({
                        "removed_props_num": removed_props_num,
                        "added_props_vector_num": "0 (because of exceeding limit)"
                    })
                    return context
                if to_update_remove:
                    update_remove_prop_values = [prop_set for _, prop_set in to_update_remove]
                    removed_num = self.prop_index.remove(update_remove_prop_values)
                    self.prop_index.to_index_file(self.index_dir_prop)
                    log.info("In to_update: Removed %s outdated property set", removed_num)
                added_props_embeddings = asyncio.run(self._get_embeddings_parallel(add_prop_values))
                self.prop_index.add(added_props_embeddings, add_propsets)
                log.info("Added %s new or updated property embeddings", len(added_props_embeddings))
                self.prop_index.to_index_file(self.index_dir_prop)
            else:
                log.debug("No update props to build vector index.")
            context.update({
                "removed_props_num": removed_props_num,
                "added_props_vector_num": len(to_add)
            })

        return context
