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
from typing import Dict, Any, List

from hugegraph_llm.config import resource_path, llm_settings, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.embedding_utils import get_embeddings_parallel, get_filename_prefix, get_index_folder_name


# FIXME: we need keep the logic same with build_semantic_index.py
class BuildGremlinExampleIndex:
    def __init__(self, embedding: BaseEmbedding, examples: List[Dict[str, str]]):
        self.folder_name = get_index_folder_name(huge_settings.graph_name, huge_settings.graph_space)
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "gremlin_examples"))
        self.examples = examples
        self.embedding = embedding
        self.filename_prefix = get_filename_prefix(llm_settings.embedding_type, getattr(embedding, "model_name", None))

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        embed_dim = 0

        if len(self.examples) > 0:
            # Use the new async parallel embedding approach from upstream
            queries = [example["query"] for example in self.examples]
            # TODO: refactor function chain async to avoid blocking
            examples_embedding = asyncio.run(get_embeddings_parallel(self.embedding, queries))
            embed_dim = len(examples_embedding[0])

            vector_index = VectorIndex(embed_dim)
            vector_index.add(examples_embedding, self.examples)
            vector_index.to_index_file(self.index_dir, self.filename_prefix)

        context["embed_dim"] = embed_dim
        return context
