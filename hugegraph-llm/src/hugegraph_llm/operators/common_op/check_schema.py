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


from typing import Any


class CheckSchema:
    def __init__(self, data):
        self.result = None
        self.data = data

    def run(self, schema=None) -> Any:
        schema = self.data or schema
        if not isinstance(schema, dict):
            raise ValueError("Input data is not a dictionary.")
        if "vertices" not in schema or "edges" not in schema:
            raise ValueError("Input data does not contain 'vertices' or 'edges'.")
        if not isinstance(schema["vertices"], list) or not isinstance(schema["edges"], list):
            raise ValueError("'vertices' or 'edges' in input data is not a list.")
        for vertex in schema["vertices"]:
            if not isinstance(vertex, dict):
                raise ValueError("Vertex in input data is not a dictionary.")
            if "vertex_label" not in vertex:
                raise ValueError("Vertex in input data does not contain 'vertex_label'.")
            if not isinstance(vertex["vertex_label"], str):
                raise ValueError("'vertex_label' in vertex is not of correct type.")
        for edge in schema["edges"]:
            if not isinstance(edge, dict):
                raise ValueError("Edge in input data is not a dictionary.")
            if (
                "edge_label" not in edge
                or "source_vertex_label" not in edge
                or "target_vertex_label" not in edge
            ):
                raise ValueError(
                    "Edge in input data does not contain "
                    "'edge_label', 'source_vertex_label', 'target_vertex_label'."
                )
            if (
                not isinstance(edge["edge_label"], str)
                or not isinstance(edge["source_vertex_label"], str)
                or not isinstance(edge["target_vertex_label"], str)
            ):
                raise ValueError(
                    "'edge_label', 'source_vertex_label', 'target_vertex_label' "
                    "in edge is not of correct type."
                )
        return schema
