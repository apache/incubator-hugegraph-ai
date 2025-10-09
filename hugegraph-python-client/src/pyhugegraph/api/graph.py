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

import json
from typing import Optional, List

from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.structure.edge_data import EdgeData
from pyhugegraph.structure.vertex_data import VertexData
from pyhugegraph.utils import huge_router as router
from pyhugegraph.utils.exceptions import NotFoundError


class GraphManager(HugeParamsBase):

    @router.http("POST", "graph/vertices")
    def addVertex(self, label, properties, id=None):
        data = {}
        if id is not None:
            data["id"] = id
        data["label"] = label
        data["properties"] = properties
        if response := self._invoke_request(data=json.dumps(data)):
            return VertexData(response)
        return None

    @router.http("POST", "graph/vertices/batch")
    def addVertices(self, input_data):
        data = []
        for item in input_data:
            data.append({"label": item[0], "properties": item[1]})
        if response := self._invoke_request(data=json.dumps(data)):
            return [VertexData({"id": item}) for item in response]
        return None

    @router.http("PUT", 'graph/vertices/"{vertex_id}"?action=append')
    def appendVertex(self, vertex_id, properties):  # pylint: disable=unused-argument
        data = {"properties": properties}
        if response := self._invoke_request(data=json.dumps(data)):
            return VertexData(response)
        return None

    @router.http("PUT", 'graph/vertices/"{vertex_id}"?action=eliminate')
    def eliminateVertex(self, vertex_id, properties):  # pylint: disable=unused-argument
        data = {"properties": properties}
        if response := self._invoke_request(data=json.dumps(data)):
            return VertexData(response)
        return None

    @router.http("GET", 'graph/vertices/"{vertex_id}"')
    def getVertexById(self, vertex_id):  # pylint: disable=unused-argument
        if response := self._invoke_request():
            return VertexData(response)
        return None

    def getVertexByPage(self, label, limit, page=None, properties=None):
        path = "graph/vertices?"
        para = ""
        para = para + "&label=" + label
        if properties:
            para = para + "&properties=" + json.dumps(properties)
        if page:
            para += f"&page={page}"
        else:
            para += "&page"
        para = para + "&limit=" + str(limit)
        path = path + para[1:]
        if response := self._sess.request(path):
            res = [VertexData(item) for item in response["vertices"]]
            next_page = response["page"]
            return res, next_page
        return None, None

    def getVertexByCondition(self, label="", limit=0, page=None, properties=None):
        path = "graph/vertices?"
        para = ""
        if label:
            para = para + "&label=" + label
        if properties:
            para = para + "&properties=" + json.dumps(properties)
        if limit > 0:
            para = para + "&limit=" + str(limit)
        if page:
            para += f"&page={page}"
        else:
            para += "&page"
        path = path + para[1:]
        if response := self._sess.request(path):
            return [VertexData(item) for item in response["vertices"]]
        return None

    @router.http("DELETE", 'graph/vertices/"{vertex_id}"')
    def removeVertexById(self, vertex_id):  # pylint: disable=unused-argument
        return self._invoke_request()

    @router.http("POST", "graph/edges")
    def addEdge(self, edge_label, out_id, in_id, properties) -> Optional[EdgeData]:
        data = {
            "label": edge_label,
            "outV": out_id,
            "inV": in_id,
            "properties": properties,
        }
        if response := self._invoke_request(data=json.dumps(data)):
            return EdgeData(response)
        return None

    @router.http("POST", "graph/edges/batch")
    def addEdges(self, input_data) -> Optional[List[EdgeData]]:
        data = []
        for item in input_data:
            data.append(
                {
                    "label": item[0],
                    "outV": item[1],
                    "inV": item[2],
                    "outVLabel": item[3],
                    "inVLabel": item[4],
                    "properties": item[5],
                }
            )
        if response := self._invoke_request(data=json.dumps(data)):
            return [EdgeData({"id": item}) for item in response]
        return None

    @router.http("PUT", "graph/edges/{edge_id}?action=append")
<<<<<<< HEAD
    def appendEdge(
        self, edge_id, properties  # pylint: disable=unused-argument
    ) -> Optional[EdgeData]:
=======
    def appendEdge(self, edge_id, properties) -> Optional[EdgeData]:  # pylint: disable=unused-argument
>>>>>>> 87ee5d3 (style: format code with black line-length 120)
        if response := self._invoke_request(data=json.dumps({"properties": properties})):
            return EdgeData(response)
        return None

    @router.http("PUT", "graph/edges/{edge_id}?action=eliminate")
<<<<<<< HEAD
    def eliminateEdge(
        self, edge_id, properties  # pylint: disable=unused-argument
    ) -> Optional[EdgeData]:
=======
    def eliminateEdge(self, edge_id, properties) -> Optional[EdgeData]:  # pylint: disable=unused-argument
>>>>>>> 87ee5d3 (style: format code with black line-length 120)
        if response := self._invoke_request(data=json.dumps({"properties": properties})):
            return EdgeData(response)
        return None

    @router.http("GET", "graph/edges/{edge_id}")
    def getEdgeById(self, edge_id) -> Optional[EdgeData]:  # pylint: disable=unused-argument
        if response := self._invoke_request():
            return EdgeData(response)
        return None

    def getEdgeByPage(
        self,
        label=None,
        vertex_id=None,
        direction=None,
        limit=0,
        page=None,
        properties=None,
    ):
        path = "graph/edges?"
        para = ""
        if vertex_id:
            if direction:
                para = para + '&vertex_id="' + vertex_id + '"&direction=' + direction
            else:
                raise NotFoundError("Direction can not be empty.")
        if label:
            para = para + "&label=" + label
        if properties:
            para = para + "&properties=" + json.dumps(properties)
        if page:
            para += f"&page={page}"
        else:
            para += "&page"
        if limit > 0:
            para = para + "&limit=" + str(limit)
        path = path + para[1:]
        if response := self._sess.request(path):
            return [EdgeData(item) for item in response["edges"]], response["page"]
        return None, None

    @router.http("DELETE", "graph/edges/{edge_id}")
    def removeEdgeById(self, edge_id) -> dict:  # pylint: disable=unused-argument
        return self._invoke_request()

    def getVerticesById(self, vertex_ids) -> Optional[List[VertexData]]:
        if not vertex_ids:
            return []
        path = "traversers/vertices?"
        for vertex_id in vertex_ids:
            path += f'ids="{vertex_id}"&'  # pylint: disable=consider-using-join
        path = path.rstrip("&")
        if response := self._sess.request(path):
            return [VertexData(item) for item in response["vertices"]]
        return None

    def getEdgesById(self, edge_ids) -> Optional[List[EdgeData]]:
        if not edge_ids:
            return []
        path = "traversers/edges?"
        for vertex_id in edge_ids:
            path += f"ids={vertex_id}&"  # pylint: disable=consider-using-join
        path = path.rstrip("&")
        if response := self._sess.request(path):
            return [EdgeData(item) for item in response["edges"]]
        return None
