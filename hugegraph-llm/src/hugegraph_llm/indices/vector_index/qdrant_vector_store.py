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

from typing import Any, List, Set, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

from hugegraph_llm.config import index_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.utils.log import log

COLLECTION_NAME_PREFIX = "hugegraph_llm_"


class QdrantVectorIndex(VectorStoreBase):
    def __init__(self, name: str, host: str, port: int, api_key=None, embed_dim: int = 1024):
        self.embed_dim = embed_dim
        self.host = host
        self.port = port
        self.name = COLLECTION_NAME_PREFIX + name
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if self.name not in collection_names:
            self._create_collection()

    def _create_collection(self):
        """Create a new collection in Qdrant."""
        self.client.create_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(size=self.embed_dim, distance=models.Distance.COSINE),
        )
        log.info("Created Qdrant collection '%s'", self.name)

    def to_index_file(self, name: str):
        # nothing to do when qdrant
        pass

    def add(self, vectors: List[List[float]], props: List[Any]):
        if len(vectors) == 0:
            return

        points = []

        for i, (vector, prop) in enumerate(zip(vectors, props)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={"property": prop},
                )
            )

        self.client.upsert(collection_name=self.name, points=points, wait=True)

    def remove(self, props: Union[Set[Any], List[Any]]) -> int:
        if isinstance(props, list):
            props = set(props)

        remove_num = 0

        for prop in props:
            serialized_prop = prop
            search_result = self.client.scroll(
                collection_name=self.name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="property",
                            match=models.MatchValue(value=serialized_prop),
                        )
                    ]
                ),
                limit=1000,
            )
            if search_result and search_result[0]:
                point_ids = [point.id for point in search_result[0]]

                if point_ids:
                    _ = self.client.delete(
                        collection_name=self.name,
                        points_selector=models.PointIdsList(points=point_ids),
                        wait=True,
                    )
                    remove_num += len(point_ids)

        return remove_num

    def search(self, query_vector: List[float], top_k: int = 5, dis_threshold: float = 0.9):
        search_result = self.client.search(collection_name=self.name, query_vector=query_vector, limit=top_k)

        result_properties = []

        for hit in search_result:
            distance = 1.0 - hit.score
            if distance < dis_threshold:
                if hit.payload is not None:
                    result_properties.append(hit.payload.get("property"))
                    log.debug("[✓] Add valid distance %s to results.", distance)
                else:
                    log.debug("[x] Hit payload is None, skipping.")
            else:
                log.debug(
                    "[x] Distance %s >= threshold %s, ignore this result.",
                    distance,
                    dis_threshold,
                )

        return result_properties

    @staticmethod
    def clean(name: str):
        client = QdrantClient(
            host=index_settings.qdrant_host, port=index_settings.qdrant_port, api_key=index_settings.qdrant_api_key
        )
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        name = COLLECTION_NAME_PREFIX + name
        if name in collection_names:
            client.delete_collection(collection_name=name)

    @staticmethod
    def from_name(name: str) -> "QdrantVectorIndex":
        assert index_settings.qdrant_host, "Qdrant host is not configured"
        return QdrantVectorIndex(
            name=COLLECTION_NAME_PREFIX + name,
            host=index_settings.qdrant_host,
            port=index_settings.qdrant_port,
            api_key=index_settings.qdrant_api_key,
        )
