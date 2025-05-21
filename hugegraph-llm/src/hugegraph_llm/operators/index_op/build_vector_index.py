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


<<<<<<< HEAD
import asyncio
import os
from typing import Dict, Any
=======
from typing import Any, Dict
>>>>>>> 38dce0b (feat(llm): vector db finished)

<<<<<<< HEAD
from hugegraph_llm.config import huge_settings, resource_path, llm_settings
from hugegraph_llm.indices.vector_index import VectorIndex
=======
from tqdm import tqdm
<<<<<<< HEAD
from hugegraph_llm.config import huge_settings, resource_path
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
=======

from hugegraph_llm.config import huge_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
>>>>>>> 38dce0b (feat(llm): vector db finished)
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.embedding_utils import (
    get_embeddings_parallel,
    get_filename_prefix,
    get_index_folder_name,
)
from hugegraph_llm.utils.log import log


class BuildVectorIndex:
    def __init__(self, embedding: BaseEmbedding, vector_index: type[VectorStoreBase]):
        self.embedding = embedding
<<<<<<< HEAD
<<<<<<< HEAD
        self.folder_name = get_index_folder_name(
            huge_settings.graph_name, huge_settings.graph_space
        )
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "chunks"))
        self.filename_prefix = get_filename_prefix(
            llm_settings.embedding_type, getattr(self.embedding, "model_name", None)
        )
        self.vector_index = VectorIndex.from_index_file(self.index_dir, self.filename_prefix)
=======
        self.index_dir = str(os.path.join(resource_path, huge_settings.graph_name, "chunks"))
        self.vector_index = FaissVectorIndex.from_name(self.index_dir)
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
=======
        self.vector_index = vector_index.from_name(
            embedding.get_embedding_dim(),
            huge_settings.graph_name,
            "chunks",
        )
>>>>>>> 38dce0b (feat(llm): vector db finished)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if "chunks" not in context:
            raise ValueError("chunks not found in context.")
        chunks = context["chunks"]
        chunks_embedding = []
        log.debug("Building vector index for %s chunks...", len(context["chunks"]))
        # TODO: use async_get_texts_embedding instead of single sync method
        chunks_embedding = asyncio.run(get_embeddings_parallel(self.embedding, chunks))
        if len(chunks_embedding) > 0:
            self.vector_index.add(chunks_embedding, chunks)
<<<<<<< HEAD
            self.vector_index.to_index_file(self.index_dir, self.filename_prefix)
=======
            self.vector_index.save_index_by_name(huge_settings.graph_name, "chunks")
>>>>>>> 38dce0b (feat(llm): vector db finished)
        return context
