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
from typing import List, Dict, Any, Set, Union

import faiss
import numpy as np

from hugegraph_llm.utils.log import log

INDEX_FILE_NAME = "index.faiss"
PROPERTIES_FILE_NAME = "properties.pkl"


class VectorIndex:
    """Comment"""
    def __init__(self, embed_dim: int = 1024):
        self.index = faiss.IndexFlatL2(embed_dim)
        self.properties = []

    @staticmethod
    def from_index_file(dir_path: str) -> "VectorIndex":
        index_file = os.path.join(dir_path, INDEX_FILE_NAME)
        properties_file = os.path.join(dir_path, PROPERTIES_FILE_NAME)
        if not os.path.exists(index_file) or not os.path.exists(properties_file):
            log.warning("No index file found, create a new one.")
            return VectorIndex()

        faiss_index = faiss.read_index(index_file)
        embed_dim = faiss_index.d
        with open(properties_file, "rb") as f:
            properties = pkl.load(f)
        vector_index = VectorIndex(embed_dim)
        vector_index.index = faiss_index
        vector_index.properties = properties
        return vector_index

    def to_index_file(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        index_file = os.path.join(dir_path, INDEX_FILE_NAME)
        properties_file = os.path.join(dir_path, PROPERTIES_FILE_NAME)
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

    def search(self, query_vector: List[float], top_k: int, dis_threshold: float = 0.9) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        if len(query_vector) != self.index.d:
            raise ValueError("Query vector dimension does not match index dimension!")

        distances, indices = self.index.search(np.array([query_vector]), top_k)
        results = []
        for dist, i in zip(distances[0], indices[0]):
            if dist < dis_threshold: # Smaller distances indicate higher similarity
                results.append(deepcopy(self.properties[i]))
                log.debug("[✓] Add valid distance %s to results.", dist)
            else:
                log.debug("[x] Distance %s ≥ threshold %s, ignore this result.", dist, dis_threshold)
        return results

    @staticmethod
    def clean(dir_path: str):
        index_file = os.path.join(dir_path, INDEX_FILE_NAME)
        properties_file = os.path.join(dir_path, PROPERTIES_FILE_NAME)
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(properties_file):
            os.remove(properties_file)
