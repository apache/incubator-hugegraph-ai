import json
from typing import List, Any, Set, Union

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)

from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.utils.log import log
from hugegraph_llm.config import index_settings

COLLECTION_NAME_PREFIX = "hugegraph_llm_"


class MilvusVectorIndex(VectorStoreBase):
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        user="",
        password="",
        embed_dim: int = 1024,
    ):
        self.embed_dim = embed_dim
        self.host = host
        self.port = port
        self.name = COLLECTION_NAME_PREFIX + name
        connections.connect(host=host, port=port, user=user, password=password)

        if not utility.has_collection(self.name):
            self._create_collection()

        self.collection = Collection(self.name)

    def _create_collection(self):
        """Create a new collection in Milvus."""
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim)
        property_field = FieldSchema(name="property", dtype=DataType.VARCHAR, max_length=65535)
        original_id_field = FieldSchema(name="original_id", dtype=DataType.INT64)

        schema = CollectionSchema(
            fields=[id_field, vector_field, property_field, original_id_field],
            description="Vector index collection",
        )

        collection = Collection(name=self.name, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    def to_index_file(self, name: str):
        self.collection.flush()

    def _deserialize_property(self, prop_str):
        """Deserialize property from JSON string."""
        try:
            return json.loads(prop_str)
        except (json.JSONDecodeError, TypeError):
            return prop_str

    def add(self, vectors: List[List[float]], props: List[Any]):
        if len(vectors) == 0:
            return

        # Get the current count to use as starting index
        count = self.collection.num_entities
        entities = []

        for i, (vector, prop) in enumerate(zip(vectors, props)):
            idx = count + i
            entities.append(
                {
                    "embedding": vector,
                    "property": prop,
                    "original_id": idx,
                }
            )

        self.collection.insert(entities)
        self.collection.flush()

    def remove(self, props: Union[Set[Any], List[Any]]) -> int:
        if isinstance(props, list):
            props = set(props)
        try:
            self.collection.load()
            remove_num = 0
            for prop in props:
                expr = f'property == "{prop}"'
                res = self.collection.delete(expr)
                if hasattr(res, "delete_count"):
                    remove_num += res.delete_count
            if remove_num > 0:
                self.collection.flush()
            return remove_num
        finally:
            self.collection.release()

    def search(self, query_vector: List[float], top_k: int, dis_threshold: float = 0.9) -> List[Any]:
        try:
            if self.collection.num_entities == 0:
                return []

            self.collection.load()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["property"],
            )

            ret = []
            for hits in results:
                for hit in hits:
                    if hit.distance < dis_threshold:
                        prop_str = hit.entity.get("property")
                        prop = self._deserialize_property(prop_str)
                        ret.append(prop)
                        log.debug("[✓] Add valid distance %s to results.", hit.distance)
                    else:
                        log.debug(
                            "[x] Distance %s >= threshold %s, ignore this result.",
                            hit.distance,
                            dis_threshold,
                        )

            return ret

        finally:
            self.collection.release()

    @staticmethod
    def clean(name: str):
        connections.connect(
            host=index_settings.milvus_host,
            port=index_settings.milvus_port,
            user=index_settings.milvus_user,
            password=index_settings.milvus_password,
        )
        if utility.has_collection(COLLECTION_NAME_PREFIX + name):
            utility.drop_collection(COLLECTION_NAME_PREFIX + name)

    @staticmethod
    def from_name(name: str) -> "MilvusVectorIndex":
        assert index_settings.milvus_host, "Qdrant host is not configured"
        return MilvusVectorIndex(
            name,
            host=index_settings.milvus_host,
            port=index_settings.milvus_port,
            user=index_settings.milvus_user,
            password=index_settings.milvus_password,
        )
