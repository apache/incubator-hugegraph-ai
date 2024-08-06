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

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union

import numpy as np


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def similarity(
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
        mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    if mode == SimilarityMode.EUCLIDEAN:
        # Using - Euclidean distance as similarity to achieve the same ranking order
        return -float(np.linalg.norm(embedding1 - embedding2))
    if mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm


class BaseEmbedding(ABC):
    """Embedding wrapper should take in a text and return a vector."""

    @abstractmethod
    def get_text_embedding(
            self,
            text: str
    ) -> List[float]:
        """Comment"""

    @abstractmethod
    async def async_get_text_embedding(
            self,
            text: str
    ) -> List[float]:
        """Comment"""

    @staticmethod
    def similarity(
            embedding1: Union[List[float], np.ndarray],
            embedding2: Union[List[float], np.ndarray],
            mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Get embedding similarity."""
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        if mode == SimilarityMode.EUCLIDEAN:
            # Using - Euclidean distance as similarity to achieve the same ranking order
            return -float(np.linalg.norm(embedding1 - embedding2))
        if mode == SimilarityMode.DOT_PRODUCT:
            return np.dot(embedding1, embedding2)
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm
