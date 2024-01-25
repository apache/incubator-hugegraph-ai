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

from pyhugegraph.utils.huge_requests import HugeSession
from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.structure.vertex_data import VertexData
from pyhugegraph.structure.edge_data import EdgeData
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
    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.session = self.set_session(HugeSession.new_session())

    def set_session(self, session):
        self.session = session
        return session

    def close(self):
        if self.session:
            self.session.close()

    def addVertex(self, label, properties, id=None):
        data = {}
        if id is not None:
            data["id"] = id
        data["label"] = label
        data["properties"] = properties
        url = f"{self._host}/graphs/{self._graph_name}/graph/vertices"
        response = self.session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, CreateError(f"create vertex failed: {response.content}")):
            res = VertexData(json.loads(response.content))
            return res
        return None

    def addVertices(self, input_data):
        url = f"{self._host}/graphs/{self._graph_name}/graph/vertices/batch"

        data = []
        for item in input_data:
            data.append({"label": item[0], "properties": item[1]})
        response = self.session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, CreateError(f"create vertexes failed: {response.content}")):
            res = []
            for item in json.loads(response.content):
                res.append(VertexData({"id": item}))
            return res
        return None

    def appendVertex(self, vertex_id, properties):
        url = f'{self._host}/graphs/{self._graph_name}/graph/vertices/"{vertex_id}"?action=append'

        data = {"properties": properties}
        response = self.session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, UpdateError(f"append vertex failed: {response.content}")):
            res = VertexData(json.loads(response.content))
            return res
        return None

    def eliminateVertex(self, vertex_id, properties):
        url = (
            f'{self._host}/graphs/{self._graph_name}/graph/vertices/"{vertex_id}"?action=eliminate'
        )

        data = {"properties": properties}
        response = self.session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, UpdateError(f"eliminate vertex failed: {response.content}")):
            res = VertexData(json.loads(response.content))
            return res
        return None

    def getVertexById(self, vertex_id):
        url = f'{self._host}/graphs/{self._graph_name}/graph/vertices/"{vertex_id}"'

        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(f"Vertex not found: {response.content}")):
            res = VertexData(json.loads(response.content))
            return res
        return None

    def getVertexByPage(self, label, limit, page=None, properties=None):
        url = f"{self._host}/graphs/{self._graph_name}/graph/vertices?"

        para = ""
        para = para + "&label=" + label
        if properties:
            para = para + "&properties=" + json.dumps(properties)
        if page:
            para += f"&page={page}"
        else:
            para += "&page"
        para = para + "&limit=" + str(limit)
        url = url + para[1:]
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(f"Vertex not found: {response.content}")):
            res = []
            for item in json.loads(response.content)["vertices"]:
                res.append(VertexData(item))
            next_page = json.loads(response.content)["page"]
            return res, next_page
        return None

    def getVertexByCondition(self, label="", limit=0, page=None, properties=None):
        url = f"{self._host}/graphs/{self._graph_name}/graph/vertices?"

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
        url = url + para[1:]
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(f"Vertex not found: {response.content}")):
            res = []
            for item in json.loads(response.content)["vertices"]:
                res.append(VertexData(item))
            return res
        return None

    def removeVertexById(self, vertex_id):
        url = f'{self._host}/graphs/{self._graph_name}/graph/vertices/"{vertex_id}"'
        response = self.session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, RemoveError(f"remove vertex failed: {response.content}")):
            return response.content
        return None

    def addEdge(self, edge_label, out_id, in_id, properties):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges"

        data = {
            "label": edge_label,
            "outV": out_id,
            "inV": in_id,
            "properties": properties,
        }
        response = self.session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, CreateError(f"created edge failed: {response.content}")):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    def addEdges(self, input_data):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges/batch"

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
        response = self.session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, CreateError(f"created edges failed:  {response.content}")):
            res = []
            for item in json.loads(response.content):
                res.append(EdgeData({"id": item}))
            return res
        return None

    def appendEdge(self, edge_id, properties):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges/{edge_id}?action=append"

        data = {"properties": properties}
        response = self.session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, UpdateError(f"append edge failed: {response.content}")):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    def eliminateEdge(self, edge_id, properties):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges/{edge_id}?action=eliminate"

        data = {"properties": properties}
        response = self.session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, UpdateError(f"eliminate edge failed: {response.content}")):
            res = EdgeData(json.loads(response.content))
            return res
        return None

    def getEdgeById(self, edge_id):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges/{edge_id}"

        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(f"not found edge: {response.content}")):
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
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges?"

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
        url = url + para[1:]
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(f"not found edges: {response.content}")):
            res = []
            for item in json.loads(response.content)["edges"]:
                res.append(EdgeData(item))
            return res, json.loads(response.content)["page"]
        return None

    def removeEdgeById(self, edge_id):
        url = f"{self._host}/graphs/{self._graph_name}/graph/edges/{edge_id}"

        response = self.session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, RemoveError(f"remove edge failed: {response.content}")):
            return response.content
        return None

    def getVerticesById(self, vertex_ids):
        if not vertex_ids:
            return []
        url = f"{self._host}/graphs/{self._graph_name}/traversers/vertices?"
        for vertex_id in vertex_ids:
            url += f'ids="{vertex_id}"&'
        url = url.rstrip("&")
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
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
        url = f"{self._host}/graphs/{self._graph_name}/traversers/edges?"
        for vertex_id in edge_ids:
            url += f"ids={vertex_id}&"
        url = url.rstrip("&")
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if response.status_code == 200 and check_if_authorized(response):
            res = []
            for item in json.loads(response.content)["edges"]:
                res.append(EdgeData(item))
            return res
        create_exception(response.content)
        return None
