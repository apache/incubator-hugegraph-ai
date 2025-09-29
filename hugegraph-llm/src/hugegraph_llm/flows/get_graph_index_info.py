#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os

from hugegraph_llm.config import huge_settings, llm_settings, resource_path
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.init_embedding import model_map
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.nodes.hugegraph_node.fetch_graph_data import FetchGraphDataNode
from PyCGraph import GPipeline
from hugegraph_llm.utils.embedding_utils import (
    get_filename_prefix,
    get_index_folder_name,
)


class GetGraphIndexInfoFlow(BaseFlow):
    def __init__(self):
        pass

    def prepare(self, prepared_input: WkFlowInput, *args, **kwargs):
        return

    def build_flow(self, *args, **kwargs):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, *args, **kwargs)
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")
        fetch_node = FetchGraphDataNode()
        pipeline.registerGElement(fetch_node, set(), "fetch_node")
        return pipeline

    def post_deal(self, pipeline=None):
        graph_summary_info = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        folder_name = get_index_folder_name(huge_settings.graph_name, huge_settings.graph_space)
        index_dir = str(os.path.join(resource_path, folder_name, "graph_vids"))
        filename_prefix = get_filename_prefix(
            llm_settings.embedding_type,
            model_map.get(llm_settings.embedding_type, None),
        )
        try:
            vector_index = VectorIndex.from_index_file(index_dir, filename_prefix)
        except FileNotFoundError:
            return json.dumps(graph_summary_info, ensure_ascii=False, indent=2)
        graph_summary_info["vid_index"] = {
            "embed_dim": vector_index.index.d,
            "num_vectors": vector_index.index.ntotal,
            "num_vids": len(vector_index.properties),
        }
        return json.dumps(graph_summary_info, ensure_ascii=False, indent=2)
