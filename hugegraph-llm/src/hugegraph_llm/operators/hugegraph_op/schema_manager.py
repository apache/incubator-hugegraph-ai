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
from typing import Any, Dict, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings


class SchemaManager:
    def __init__(self, graph_name: str):
        self.graph_name = graph_name
        self.client = PyHugeClient(
            url=huge_settings.graph_url,
            graph=self.graph_name,
            user=huge_settings.graph_user,
            pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )
        self.schema = self.client.schema()

    def simple_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        mini_schema = {}  # type: ignore

        # Add necessary vertexlabels items (3)
        if "vertexlabels" in schema:
            mini_schema["vertexlabels"] = []
            for vertex in schema["vertexlabels"]:
                new_vertex = {key: vertex[key] for key in ["id", "name", "properties"] if key in vertex}
                mini_schema["vertexlabels"].append(new_vertex)

        # Add necessary edgelabels items (4)
        if "edgelabels" in schema:
            mini_schema["edgelabels"] = []
            for edge in schema["edgelabels"]:
                new_edge = {
                    key: edge[key] for key in ["name", "source_label", "target_label", "properties"] if key in edge
                }
                mini_schema["edgelabels"].append(new_edge)

        return mini_schema

    def run(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if context is None:
            context = {}
        schema = self.schema.getSchema()
        if not schema["vertexlabels"] and not schema["edgelabels"]:
            raise Exception(f"Can not get {self.graph_name}'s schema from HugeGraph!")

        context.update({"schema": schema})
        # TODO: enhance the logic here
        context["simple_schema"] = self.simple_schema(schema)
        return context
