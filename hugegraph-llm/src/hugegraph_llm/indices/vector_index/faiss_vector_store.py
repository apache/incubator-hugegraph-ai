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

import os
import pickle as pkl
from copy import deepcopy
from typing import Any, Dict, List, Set, Union

import faiss
import numpy as np

from hugegraph_llm.config import resource_path
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.utils.log import log

INDEX_FILE_NAME = "index.faiss"
PROPERTIES_FILE_NAME = "properties.pkl"


class FaissVectorIndex(VectorStoreBase):
    def __init__(self, embed_dim: int = 1024):
        self.index = faiss.IndexFlatL2(embed_dim)
        self.properties: list[Any] = []

<<<<<<< HEAD
        

    def to_index_file(self, dir_path: str, filename_prefix: str = None):
        """Save index to files, supporting model-specific filenames."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        index_name = f"{filename_prefix}_{INDEX_FILE_NAME}" if filename_prefix else INDEX_FILE_NAME
        property_name = (
            f"{filename_prefix}_{PROPERTIES_FILE_NAME}" if filename_prefix else PROPERTIES_FILE_NAME
        )
        index_file = os.path.join(dir_path, index_name)
        properties_file = os.path.join(dir_path, property_name)
=======
    def save_index_by_name(self, *name: str):
        os.makedirs(os.path.join(resource_path, *name), exist_ok=True)
        index_file = os.path.join(resource_path, *name, INDEX_FILE_NAME)
        properties_file = os.path.join(resource_path, *name, PROPERTIES_FILE_NAME)
>>>>>>> 38dce0b (feat(llm): vector db finished)
        faiss.write_index(self.index, index_file)
        with open(properties_file, "wb") as f:
            pkl.dump(self.properties, f)

    def add(self, vectors: List[List[float]], props: List[Any]):
        if len(vectors) == 0:
            return
        if self.index.ntotal == 0 and len(vectors[0]) != self.index.d:
            self.index = faiss.IndexFlatL2(len(vectors[0]))
        self.index.add(np.array(vectors))
        self.properties.extend(props)

    def remove(self, props: Union[Set[Any], List[Any]]) -> int:
        if isinstance(props, list):
            props = set(props)
        indices = []
        remove_num = 0

        for i, p in enumerate(self.properties):
            if p in props:
                indices.append(i)
                remove_num += 1
        self.index.remove_ids(np.array(indices))
        self.properties = [p for i, p in enumerate(self.properties) if i not in indices]
        return remove_num

    def search(
        self, query_vector: List[float], top_k: int, dis_threshold: float = 0.9
    ) -> List[Any]:
        if self.index.ntotal == 0:
            return []

        if len(query_vector) != self.index.d:
            raise ValueError("Query vector dimension does not match index dimension!")

        distances, indices = self.index.search(np.array([query_vector]), top_k)
        results = []
        for dist, i in zip(distances[0], indices[0]):
            if dist < dis_threshold:
                results.append(deepcopy(self.properties[i]))
                log.debug("[✓] Add valid distance %s to results.", dist)
            else:
                log.debug(
                    "[x] Distance %s >= threshold %s, ignore this result.",
                    dist,
                    dis_threshold,
                )
        return results

    def get_all_properties(self) -> list[Any]:
        return self.properties

    def get_vector_index_info(
        self,
    ) -> Dict:
        return {
            "embed_dim": self.index.d,
            "vector_info": {
                "chunk_vector_num": self.index.ntotal,
                "graph_vid_vector_num": self.index.ntotal,
                "graph_properties_vector_num": len(self.properties),
            },
        }

    @staticmethod
<<<<<<< HEAD
    def from_index_file(
        dir_path: str, filename_prefix: str | None = None, record_miss: bool = True
    ) -> "FaissVectorIndex":
        """Load index from files, supporting model-specific filenames.

        If prefixed files are missing, optionally warn and return an empty index.
        Also validates vector/property count consistency.
        """
        index_name = f"{filename_prefix}_{INDEX_FILE_NAME}" if filename_prefix else INDEX_FILE_NAME
        property_name = (
            f"{filename_prefix}_{PROPERTIES_FILE_NAME}" if filename_prefix else PROPERTIES_FILE_NAME
        )
        index_file = os.path.join(dir_path, index_name)
        properties_file = os.path.join(dir_path, property_name)
        missing = [p for p in [index_file, properties_file] if not os.path.exists(p)]
        if missing:
            if record_miss:
                log.warning(
                    "Missing vector files: %s. Need create a new one for it.", ", ".join(missing)
                )
            return FaissVectorIndex()

        try:
            faiss_index = faiss.read_index(index_file)
            with open(properties_file, "rb") as f:
                properties = pkl.load(f)
        except (RuntimeError, pkl.UnpicklingError, OSError) as e:  # pragma: no cover
            log.error(
                "Failed to load index files for model '%s': %s",
                filename_prefix or "default",
                e,
            )
            raise RuntimeError(
                f"Could not load index files for model '{filename_prefix or 'default'}'. "
                f"Original error ({type(e).__name__}): {e}"
            ) from e

        if faiss_index.ntotal != len(properties):
            raise RuntimeError(
                f"Data inconsistency: index for model '{filename_prefix or 'default'}' has "
                f"{faiss_index.ntotal} vectors, but {len(properties)} properties."
            )

        embed_dim = faiss_index.d
        vector_index = FaissVectorIndex(embed_dim)
        vector_index.index = faiss_index
        vector_index.properties = properties
        return vector_index

    @staticmethod
    def clean(dir_path: str, filename_prefix: str = None):
        """Clean index files, supporting model-specific filenames.

        This method deletes the index and properties files associated with a specific model.
        If model_name is None, it targets the default files.
        """
        index_name = f"{filename_prefix}_{INDEX_FILE_NAME}" if filename_prefix else INDEX_FILE_NAME
        property_name = (
            f"{filename_prefix}_{PROPERTIES_FILE_NAME}" if filename_prefix else PROPERTIES_FILE_NAME
        )
        index_file = os.path.join(dir_path, index_name)
        properties_file = os.path.join(dir_path, property_name)

        for file in [index_file, properties_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    log.info("Removed index file: %s", file)
                except OSError as e:
                    log.error("Error removing file %s: %s", file, e)
        
=======
    def clean(*name: str):
        index_file = os.path.join(resource_path, *name, INDEX_FILE_NAME)
        properties_file = os.path.join(resource_path, *name, PROPERTIES_FILE_NAME)
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(properties_file):
            os.remove(properties_file)
>>>>>>> 38dce0b (feat(llm): vector db finished)

    @staticmethod
    def from_name(embed_dim: int, *name: str) -> "FaissVectorIndex":
        index_file = os.path.join(resource_path, *name, INDEX_FILE_NAME)
        properties_file = os.path.join(resource_path, *name, PROPERTIES_FILE_NAME)
        if not os.path.exists(index_file) or not os.path.exists(properties_file):
            log.warning("No index file found, create a new one.")
            return FaissVectorIndex(embed_dim)

        faiss_index = faiss.read_index(index_file)
        with open(properties_file, "rb") as f:
            properties = pkl.load(f)
        vector_index = FaissVectorIndex(embed_dim)
        if faiss_index.d == vector_index.index.d:
            # when dim same, use old
            vector_index.index = faiss_index
            vector_index.properties = properties
        else:
            log.warning("dim is different, create a new one.")
        return vector_index

    @staticmethod
    def exist(*name: str) -> bool:
        index_file = os.path.join(resource_path, *name, INDEX_FILE_NAME)
        properties_file = os.path.join(resource_path, *name, PROPERTIES_FILE_NAME)
        return os.path.exists(index_file) and os.path.exists(properties_file)
