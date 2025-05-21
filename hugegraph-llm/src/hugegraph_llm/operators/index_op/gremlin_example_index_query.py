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


import asyncio
import os
from typing import Any, Dict, List, Optional

import pandas as pd

<<<<<<< HEAD
<<<<<<< HEAD
from hugegraph_llm.config import resource_path, llm_settings, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex, INDEX_FILE_NAME, PROPERTIES_FILE_NAME
=======
from hugegraph_llm.config import resource_path, huge_settings, llm_settings
=======
from hugegraph_llm.config import resource_path
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
>>>>>>> 38dce0b (feat(llm): vector db finished)
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.utils.embedding_utils import (
    get_embeddings_parallel,
    get_filename_prefix,
    get_index_folder_name,
)
from hugegraph_llm.utils.log import log


class GremlinExampleIndexQuery:
    def __init__(
        self, vector_index: type[VectorStoreBase], embedding: Optional[BaseEmbedding] = None, num_examples: int = 1
    ):
        self.embedding = embedding or Embeddings().get_embedding()
        self.num_examples = num_examples
<<<<<<< HEAD
        self.folder_name = get_index_folder_name(
            huge_settings.graph_name, huge_settings.graph_space
        )
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "gremlin_examples"))
        self.filename_prefix = get_filename_prefix(
            llm_settings.embedding_type, getattr(self.embedding, "model_name", None)
        )
        self._ensure_index_exists()
        self.vector_index = FaissVectorIndex.from_index_file(self.index_dir, self.filename_prefix)

    def _ensure_index_exists(self):
        index_name = (
            f"{self.filename_prefix}_{INDEX_FILE_NAME}" if self.filename_prefix else INDEX_FILE_NAME
        )
        props_name = (
            f"{self.filename_prefix}_{PROPERTIES_FILE_NAME}"
            if self.filename_prefix
            else PROPERTIES_FILE_NAME
        )
        if not (
            os.path.exists(os.path.join(self.index_dir, index_name))
            and os.path.exists(os.path.join(self.index_dir, props_name))
        ):
=======
        if not vector_index.exist("gremlin_examples"):
>>>>>>> 38dce0b (feat(llm): vector db finished)
            log.warning("No gremlin example index found, will generate one.")
            self.vector_index = vector_index.from_name(self.embedding.get_embedding_dim(), "gremlin_examples")

            self._build_default_example_index()
        else:
            self.vector_index = vector_index.from_name(self.embedding.get_embedding_dim(), "gremlin_examples")

    def _get_match_result(self, context: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        if self.num_examples <= 0:
            return []

        query_embedding = context.get("query_embedding")
        if not isinstance(query_embedding, list):
            query_embedding = self.embedding.get_texts_embeddings([query])[0]
        return self.vector_index.search(query_embedding, self.num_examples, dis_threshold=1.8)

    def _build_default_example_index(self):
<<<<<<< HEAD
        properties = pd.read_csv(os.path.join(resource_path, "demo", "text2gremlin.csv")).to_dict(
            orient="records"
        )
        # TODO: reuse the logic in build_semantic_index.py (consider extract the batch-embedding method)
        queries = [row["query"] for row in properties]
        embeddings = asyncio.run(get_embeddings_parallel(self.embedding, queries))
        vector_index = FaissVectorIndex(len(embeddings[0]))
=======
        properties = pd.read_csv(os.path.join(resource_path, "demo", "text2gremlin.csv")).to_dict(orient="records")
        from concurrent.futures import ThreadPoolExecutor

<<<<<<< HEAD
        # TODO: use asyncio for IO tasks
=======
        # TODO: reuse the logic in build_semantic_index.py (consider extract the batch-embedding method)
>>>>>>> 38dce0b (feat(llm): vector db finished)
        with ThreadPoolExecutor() as executor:
            embeddings = list(
                tqdm(
                    executor.map(self.embedding.get_text_embedding, [row["query"] for row in properties]),
                    total=len(properties),
                )
            )
<<<<<<< HEAD
        vector_index = FaissVectorIndex(len(embeddings[0]))
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
        vector_index.add(embeddings, properties)
        vector_index.to_index_file(self.index_dir, self.filename_prefix)
=======
        self.vector_index.add(embeddings, properties)
        self.vector_index.save_index_by_name("gremlin_examples")
>>>>>>> 38dce0b (feat(llm): vector db finished)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query")
        if not query:
            raise ValueError("query is required")

        context["match_result"] = self._get_match_result(context, query)
        return context
