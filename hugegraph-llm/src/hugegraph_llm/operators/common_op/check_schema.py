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


from typing import Any, Optional, Dict
from hugegraph_llm.utils.log import log


class CheckSchema:
    def __init__(self, data):
        self.result = None
        self.data = data

    def run(self, context: Optional[Dict[str, Any]] = None) -> Any:  # pylint: disable=too-many-branches
        if context is None:
            context = {}
        schema = self.data or context.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("Input data is not a dictionary.")
        if "vertexlabels" not in schema or "edgelabels" not in schema:
            raise ValueError("Input data does not contain 'vertexlabels' or 'edgelabels'.")
        if not isinstance(schema["vertexlabels"], list) or not isinstance(schema["edgelabels"], list):
            raise ValueError("'vertexlabels' or 'edgelabels' in input data is not a list.")
        for vertex in schema["vertexlabels"]:
            if not isinstance(vertex, dict):
                raise ValueError("Vertex in input data is not a dictionary.")
            if "name" not in vertex:
                raise ValueError("Vertex in input data does not contain 'name'.")
            if not isinstance(vertex["name"], str):
                raise ValueError("'name' in vertex is not of correct type.")
            if "properties" not in vertex:
                raise ValueError("Vertex in input data does not contain 'properties'.")
            properties = vertex["properties"]
            if not isinstance(properties, list):
                raise ValueError("'properties' in vertex is not of correct type.")
            if len(properties) == 0:
                raise ValueError("'properties' in vertex is empty.")
            primary_keys = vertex.get("primary_keys", properties[:1])
            if not isinstance(primary_keys, list):
                raise ValueError("'primary_keys' in vertex is not of correct type.")
            new_primary_keys = []
            for key in primary_keys:
                if key not in properties:
                    log.waring("Primary key '%s' not found in properties has been auto removed.", key)
                else:
                    new_primary_keys.append(key)
            if len(new_primary_keys) == 0:
                raise ValueError(f"primary keys of vertexLabel {vertex['vertex_label']} is empty.")
            vertex["primary_keys"] = new_primary_keys
            nullable_keys = vertex.get("nullable_keys", properties[1:])
            if not isinstance(nullable_keys, list):
                raise ValueError("'nullable_keys' in vertex is not of correct type.")
            new_nullable_keys = []
            for key in nullable_keys:
                if key not in properties:
                    log.warning("Nullable key '%s' not found in properties has been auto removed.", key)
                else:
                    new_nullable_keys.append(key)
            vertex["nullable_keys"] = new_nullable_keys
        for edge in schema["edgelabels"]:
            if not isinstance(edge, dict):
                raise ValueError("Edge in input data is not a dictionary.")
            if "name" not in edge or "source_label" not in edge or "target_label" not in edge:
                raise ValueError("Edge in input data does not contain " "'name', 'source_label', 'target_label'.")
            if (
                not isinstance(edge["name"], str)
                or not isinstance(edge["source_label"], str)
                or not isinstance(edge["target_label"], str)
            ):
                raise ValueError("'name', 'source_label', 'target_label' " "in edge is not of correct type.")
        context.update({"schema": schema})
        return context
