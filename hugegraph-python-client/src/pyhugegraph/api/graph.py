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

from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.structure.vertex_data import VertexData
from pyhugegraph.structure.edge_data import EdgeData
from pyhugegraph.utils import huge_router as router
from pyhugegraph.utils.exceptions import (
    NotFoundError,
    CreateError,
    RemoveError,
    UpdateError,
)
from pyhugegraph.utils.util import (
    create_exception,
    check_if_authorized,
    check_if_success,
)


class GraphManager(HugeParamsBase):

    @router.http("POST", "graph/vertices")
    def addVertex(self, label, properties, id=None):
        data = {}
        if id is not None:
            data["id"] = id
        data["label"] = label
        data["properties"] = properties
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, CreateError(f"create vertex failed: {str(response.content)}")
        ):
            res = VertexData(json.loads(response.content))
            return res
        return None

    @router.http("POST", "graph/vertices/batch")
    def addVertices(self, input_data):
        data = []
        for item in input_data:
            data.append({"label": item[0], "properties": item[1]})
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, CreateError(f"create vertexes failed: {str(response.content)}")
        ):
            res = []
            for item in json.loads(response.content):
                res.append(VertexData({"id": item}))
            return res
        return None

    @router.http("PUT", 'graph/vertices/"{vertex_id}"?action=append')
    def appendVertex(self, vertex_id, properties):  # pylint: disable=unused-argument
        data = {"properties": properties}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, UpdateError(f"append vertex failed: {str(response.content)}")
        ):
            res = VertexData(json.loads(response.content))
            return res
        return None

    @router.http("PUT", 'graph/vertices/"{vertex_id}"?action=eliminate')
    def eliminateVertex(self, vertex_id, properties):  # pylint: disable=unused-argument
        data = {"properties": properties}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, UpdateError(f"eliminate vertex failed: {str(response.content)}")
        ):
            res = VertexData(json.loads(response.content))
            return res
        return None

    @router.http("GET", 'graph/vertices/"{vertex_id}"')
    def getVertexById(self, vertex_id):  # pylint: disable=unused-argument
        response = self._invoke_request()
        if check_if_success(
            response, NotFoundError(f"Vertex not found: {str(response.content)}")
        ):
            res = VertexData(json.loads(response.content))
            return res
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
        response = self._sess.request(path)
        if check_if_success(
            response, NotFoundError(f"Vertex not found: {str(response.content)}")
        ):
            res = []
            for item in json.loads(response.content)["vertices"]:
                res.append(VertexData(item))
            next_page = json.loads(response.content)["page"]
            return res, next_page
        return None

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
        response = self._sess.request(path)
        if check_if_success(
            response, NotFoundError(f"Vertex not found: {str(response.content)}")
        ):
            res = []
            for item in json.loads(response.content)["vertices"]:
                res.append(VertexData(item))
            return res
        return None

    @router.http("DELETE", 'graph/vertices/"{vertex_id}"')
    def removeVertexById(self, vertex_id):  # pylint: disable=unused-argument
        response = self._invoke_request()
        if check_if_success(
            response, RemoveError(f"remove vertex failed: {str(response.content)}")
        ):
            return response.content
        return None

    @router.http("POST", "graph/edges")
    def addEdge(self, edge_label, out_id, in_id, properties):
        data = {
            "label": edge_label,
            "outV": out_id,
            "inV": in_id,
            "properties": properties,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, CreateError(f"created edge failed: {str(response.content)}")
        ):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    @router.http("POST", "graph/edges/batch")
    def addEdges(self, input_data):
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
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(
            response, CreateError(f"created edges failed:  {str(response.content)}")
        ):
            res = []
            for item in json.loads(response.content):
                res.append(EdgeData({"id": item}))
            return res
        return None

    @router.http("PUT", "graph/edges/{edge_id}?action=append")
    def appendEdge(self, edge_id, properties):  # pylint: disable=unused-argument
        response = self._invoke_request(data=json.dumps({"properties": properties}))
        if check_if_success(
            response, UpdateError(f"append edge failed: {str(response.content)}")
        ):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    @router.http("PUT", "graph/edges/{edge_id}?action=eliminate")
    def eliminateEdge(self, edge_id, properties):  # pylint: disable=unused-argument
        response = self._invoke_request(data=json.dumps({"properties": properties}))
        if check_if_success(
            response, UpdateError(f"eliminate edge failed: {str(response.content)}")
        ):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    @router.http("GET", "graph/edges/{edge_id}")
    def getEdgeById(self, edge_id):  # pylint: disable=unused-argument
        response = self._invoke_request()
        if check_if_success(
            response, NotFoundError(f"not found edge: {str(response.content)}")
        ):
            res = EdgeData(json.loads(response.content))
            return res
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
        path = f"graph/edges?"
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
        response = self._sess.request(path)
        if check_if_success(
            response, NotFoundError(f"not found edges: {str(response.content)}")
        ):
            res = []
            for item in json.loads(response.content)["edges"]:
                res.append(EdgeData(item))
            return res, json.loads(response.content)["page"]
        return None

    @router.http("DELETE", "graph/edges/{edge_id}")
    def removeEdgeById(self, edge_id):  # pylint: disable=unused-argument
        response = self._invoke_request()
        if check_if_success(
            response, RemoveError(f"remove edge failed: {str(response.content)}")
        ):
            return response.content
        return None

    def getVerticesById(self, vertex_ids):
        if not vertex_ids:
            return []
        path = "traversers/vertices?"
        for vertex_id in vertex_ids:
            path += f'ids="{vertex_id}"&'
        path = path.rstrip("&")
        response = self._sess.request(path)
        if response.status_code == 200 and check_if_authorized(response):
            res = []
            for item in json.loads(response.content)["vertices"]:
                res.append(VertexData(item))
            return res
        create_exception(response.content)
        return None

    def getEdgesById(self, edge_ids):
        if not edge_ids:
            return []
        path = "traversers/edges?"
        for vertex_id in edge_ids:
            path += f"ids={vertex_id}&"
        path = path.rstrip("&")
        response = self._sess.request(path)
        if response.status_code == 200 and check_if_authorized(response):
            res = []
            for item in json.loads(response.content)["edges"]:
                res.append(EdgeData(item))
            return res
        create_exception(response.content)
        return None
