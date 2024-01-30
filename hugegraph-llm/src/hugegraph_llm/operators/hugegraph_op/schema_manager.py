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


from hugegraph_llm.utils.config import Config
from hugegraph_llm.utils.constants import Constants
from pyhugegraph.client import PyHugeClient


class SchemaManager:
    def __init__(self, graph_name: str):
        config = Config(section=Constants.HUGEGRAPH_CONFIG)
        self.graph_name = graph_name
        self.client = PyHugeClient(
            config.get_graph_ip(),
            config.get_graph_port(),
            graph_name,
            config.get_graph_user(),
            config.get_graph_pwd(),
        )
        self.schema = self.client.schema()

    def run(self, data: dict):
        schema = self.schema.getSchema()
        vertices = []
        for vl in schema["vertexlabels"]:
            vertex = {"vertex_label": vl["name"], "properties": vl["properties"]}
            vertices.append(vertex)
        edges = []
        for el in schema["edgelabels"]:
            edge = {
                "edge_label": el["name"],
                "source_vertex_label": el["source_label"],
                "target_vertex_label": el["target_label"],
                "properties": el["properties"],
            }
            edges.append(edge)
        if not vertices and not edges:
            raise Exception(f"Can not get {self.graph_name}'s schema from HugeGraph!")
        return {"vertices": vertices, "edges": edges}
