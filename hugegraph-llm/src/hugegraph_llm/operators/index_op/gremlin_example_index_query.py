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
from typing import Dict, Any

from tqdm import tqdm
import pandas as pd
from hugegraph_llm.config import resource_path
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.utils.log import log


class GremlinExampleIndexQuery:
    def __init__(self, embedding: BaseEmbedding = None, num_examples: int = 1):
        self.embedding = embedding or Embeddings().get_embedding()
        self.num_examples = num_examples
        self.index_dir = os.path.join(resource_path, "gremlin_examples")
        if not (os.path.exists(os.path.join(self.index_dir, "index.faiss"))
                and os.path.exists(os.path.join(self.index_dir, "properties.pkl"))):
            log.warning("No gremlin example index found, will generate one.")
            self._build_default_example_index()
        self.vector_index = VectorIndex.from_index_file(self.index_dir)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query")
        assert query, "query is required"
        if self.num_examples == 0:
            context["match_result"] = []
        else:
            if "query_embedding" in context and isinstance(context["query_embedding"], list):
                query_embedding = context["query_embedding"]
            else:
                query_embedding = self.embedding.get_text_embedding(query)
            context["match_result"] = self.vector_index.search(query_embedding, self.num_examples, dis_threshold=2)
        return context

    def _build_default_example_index(self):
        properties = (pd.read_csv(os.path.join(resource_path, "demo", "text2gremlin.csv"))
                      .to_dict(orient="records"))
        embeddings = [self.embedding.get_text_embedding(row["query"]) for row in tqdm(properties)]
        vector_index = VectorIndex(len(embeddings[0]))
        vector_index.add(embeddings, properties)
        vector_index.to_index_file(self.index_dir)
