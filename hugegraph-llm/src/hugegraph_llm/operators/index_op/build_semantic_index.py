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

    # TODO: use asyncio for IO tasks
    def _get_embeddings_parallel(self, vids: list[str]) -> list[Any]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            embeddings = list(tqdm(executor.map(self.embedding.get_text_embedding, vids), total=len(vids)))
        return embeddings

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]: # pylint: disable=too-many-statements
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
            added_embeddings = self._get_embeddings_parallel(vids_to_process)
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
                result = self.client.gremlin().exec(gremlin=gremlin_query)["data"]
                results.extend(result)
            present_props = []
            seen = set()
            for item in results:
                vid = item["vid"]
                for key, value in item["properties"].items():
                    prop = f"{key}: {value}"
                    if prop not in seen:
                        seen.add(prop)
                        present_props.append((vid, prop))
            past_props = self.prop_index.properties
            removed_props = set(past_props) - set(present_props)
            removed_props_num = self.prop_index.remove(removed_props)
            if removed_props:
                self.prop_index.to_index_file(self.index_dir_prop)
            added_props = list(set(present_props) - set(past_props))
            if len(added_props) > 100000:
                log.warning("The number of props > 100000, please select which properties to vectorize.")
                context.update({
                    "removed_props_num": removed_props_num,
                    "added_props_vector_num": "0 (because of exceeding limit)"
                })
                return context
            if added_props:
                added_props_key = [item[1] for item in added_props]
                added_props_embeddings = self._get_embeddings_parallel(added_props_key)
                log.info("Building vector index for %s props...", len(added_props_key))
                self.prop_index.add(added_props_embeddings, added_props)
                self.prop_index.to_index_file(self.index_dir_prop)
            else:
                log.debug("No update props to build vector index.")
            context.update({
                "removed_props_num": removed_props_num,
                "added_props_vector_num": len(added_props)
            })

        return context
