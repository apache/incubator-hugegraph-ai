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


from typing import Optional, Dict, Any
import gradio as gr

from pyhugegraph.client import PyHugeClient
from hugegraph_llm.utils.log import log


class FetchGraphData:
    def __init__(self, graph: PyHugeClient):
        self.graph = graph

    def run(self, graph_summary_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        limit_vertices = 10000
        limit_edges = 100
        gr.Info(f"Returning a maximum of {limit_vertices} vertices. \n Returning a maximum of {limit_edges} edges.")
        
        if graph_summary_info is None:
            graph_summary_info = {}
        if "num_vertices" not in graph_summary_info:
            graph_summary_info["num_vertices"] = self.graph.gremlin().exec("g.V().id().count()")["data"]
        if "num_edges" not in graph_summary_info:
            graph_summary_info["num_edges"] = self.graph.gremlin().exec("g.E().id().count()")["data"]
        if "vertices" not in graph_summary_info:
            graph_summary_info["vertices"] = self.graph.gremlin().exec(f"g.V().id().limit({limit_vertices})")["data"]
        if "edges" not in graph_summary_info:
            graph_summary_info["edges"] = self.graph.gremlin().exec(f"g.E().id().limit({limit_edges})")["data"]
        return graph_summary_info
