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

from pyhugegraph.client import PyHugeClient


class FetchGraphData:
    def __init__(self, graph: PyHugeClient):
        self.graph = graph

    def run(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if context is None:
            context = {}
        if "num_vertices" not in context:
            context["num_vertices"] = self.graph.gremlin().exec("g.V().id().count()")["data"]
        if "num_edges" not in context:
            context["num_edges"] = self.graph.gremlin().exec("g.E().id().count()")["data"]
        if "vertices" not in context:
            context["vertices"] = self.graph.gremlin().exec("g.V().id().limit(10000)")["data"]
        if "edges" not in context:
            context["edges"] = self.graph.gremlin().exec("g.E().id().limit(10000)")["data"]
        return context
