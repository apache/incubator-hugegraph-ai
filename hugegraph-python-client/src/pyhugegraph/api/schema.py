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

from typing import Optional, Callable, Dict
from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.api.schema_manage.edge_label import EdgeLabel
from pyhugegraph.api.schema_manage.index_label import IndexLabel
from pyhugegraph.api.schema_manage.property_key import PropertyKey
from pyhugegraph.api.schema_manage.vertex_label import VertexLabel
from pyhugegraph.structure.edge_label_data import EdgeLabelData
from pyhugegraph.structure.index_label_data import IndexLabelData
from pyhugegraph.structure.property_key_data import PropertyKeyData
from pyhugegraph.structure.vertex_label_data import VertexLabelData
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.util import check_if_success
from pyhugegraph.utils import huge_router as router


class SchemaManager(HugeParamsBase):
    """
    create schemas, including propertyKey/vertexLabel/edgeLabel/indexLabel
    """

    def propertyKey(self, property_name) -> PropertyKey:
        property_key = PropertyKey(self._sess)
        property_key.create_parameter_holder()
        property_key.add_parameter("name", property_name)
        property_key.add_parameter("not_exist", True)
        return property_key

    def vertexLabel(self, vertex_name):
        vertex_label = VertexLabel(self._sess)
        vertex_label.create_parameter_holder()
        vertex_label.add_parameter("name", vertex_name)
        # vertex_label.add_parameter("id_strategy", "AUTOMATIC")
        vertex_label.add_parameter("not_exist", True)
        return vertex_label

    def edgeLabel(self, name):
        edge_label = EdgeLabel(self._sess)
        edge_label.create_parameter_holder()
        edge_label.add_parameter("name", name)
        edge_label.add_parameter("not_exist", True)
        return edge_label

    def indexLabel(self, name):
        index_label = IndexLabel(self._sess)
        index_label.create_parameter_holder()
        index_label.add_parameter("name", name)
        return index_label

    @router.http("GET", "schema?format={format}")
    def getSchema(self, format: str = "json") -> Optional[Dict]:
        response = self._invoke_request()
        error = NotFoundError(f"schema not found: {str(response.content)}")
        if check_if_success(response, error):
            schema = json.loads(response.content)
            return schema
        return None

    @router.http("GET", "schema/propertykeys/{property_name}")
    def getPropertyKey(self, property_name):
        response = self._invoke_request()
        error = NotFoundError(f"PropertyKey not found: {str(response.content)}")
        if check_if_success(response, error):
            property_keys_data = PropertyKeyData.from_dict(json.loads(response.content))
            return property_keys_data
        return None

    @router.http("GET", "schema/propertykeys")
    def getPropertyKeys(self):
        response = self._invoke_request()
        res = []
        if check_if_success(response):
            for item in json.loads(response.content)["propertykeys"]:
                res.append(PropertyKeyData.from_dict(item))
            return res
        return None

    @router.http("GET", "schema/vertexlabels/{name}")
    def getVertexLabel(self, name):
        response = self._invoke_request()
        error = NotFoundError(f"VertexLabel not found: {str(response.content)}")
        if check_if_success(response, error):
            res = VertexLabelData(json.loads(response.content))
            return res
        return None

    @router.http("GET", "schema/vertexlabels")
    def getVertexLabels(self):
        response = self._invoke_request()
        res = []
        if check_if_success(response):
            for item in json.loads(response.content)["vertexlabels"]:
                res.append(VertexLabelData(item))
        return res

    @router.http("GET", "schema/vertexlabels/{label_name}")
    def getEdgeLabel(self, label_name: str):
        response = self._invoke_request()
        error = NotFoundError(f"EdgeLabel not found: {str(response.content)}")
        if check_if_success(response, error):
            res = EdgeLabelData(json.loads(response.content))
            return res
        return None

    @router.http("GET", "schema/edgelabels")
    def getEdgeLabels(self):
        response = self._invoke_request()
        res = []
        if check_if_success(response):
            for item in json.loads(response.content)["edgelabels"]:
                res.append(EdgeLabelData(item))
        return res

    @router.http("GET", "schema/edgelabels")
    def getRelations(self):
        response = self._invoke_request()
        res = []
        if check_if_success(response):
            for item in json.loads(response.content)["edgelabels"]:
                res.append(EdgeLabelData(item).relations())
        return res

    @router.http("GET", "schema/indexlabels/{name}")
    def getIndexLabel(self, name):
        response = self._invoke_request()
        error = NotFoundError(f"EdgeLabel not found: {str(response.content)}")
        if check_if_success(response, error):
            res = IndexLabelData(json.loads(response.content))
            return res
        return None

    @router.http("GET", "schema/indexlabels")
    def getIndexLabels(self):
        response = self._invoke_request()
        res = []
        if check_if_success(response):
            for item in json.loads(response.content)["indexlabels"]:
                res.append(IndexLabelData(item))
        return res
