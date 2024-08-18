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


import time
from typing import Dict, Any, Optional, Literal, Union, List

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.common_op.check_schema import CheckSchema
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.hugegraph_op.fetch_graph_data import FetchGraphData
from hugegraph_llm.operators.index_op.build_semantic_index import BuildSemanticIndex
from hugegraph_llm.operators.index_op.build_vector_index import BuildVectorIndex
from hugegraph_llm.operators.document_op.chunk_split import ChunkSplit
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import CommitToKg
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.llm_op.disambiguate_data import DisambiguateData
from hugegraph_llm.operators.llm_op.info_extract import InfoExtract
from hugegraph_llm.operators.llm_op.property_graph_extract import PropertyGraphExtract
from hugegraph_llm.utils.log import log


class KgBuilder:
    def __init__(self, llm: BaseLLM, embedding: Optional[BaseEmbedding] = None,
                 graph: Optional[PyHugeClient] = None):
        self.operators = []
        self.llm = llm
        self.embedding = embedding
        self.graph = graph
        self.result = None

    def import_schema(self, from_hugegraph=None, from_extraction=None, from_user_defined=None):
        if from_hugegraph:
            self.operators.append(SchemaManager(from_hugegraph))
        elif from_user_defined:
            self.operators.append(CheckSchema(from_user_defined))
        elif from_extraction:
            raise Exception("Not implemented yet")
        else:
            raise Exception("No input data")
        return self

    def fetch_graph_data(self):
        self.operators.append(FetchGraphData(self.graph))
        return self

    def chunk_split(
            self,
            text: Union[str, List[str]],  # text to be split
            split_type: Literal["paragraph", "sentence"] = "paragraph",
            language: Literal["zh", "en"] = "zh"
    ):
        self.operators.append(ChunkSplit(text, split_type, language))
        return self

    def extract_info(self, example_prompt: Optional[str] = None,
                     extract_type: Literal["triples", "property_graph"] = "triples"):
        if extract_type == "triples":
            self.operators.append(InfoExtract(self.llm, example_prompt))
        elif extract_type == "property_graph":
            self.operators.append(PropertyGraphExtract(self.llm, example_prompt))
        return self

    def disambiguate_word_sense(self):
        self.operators.append(DisambiguateData(self.llm))
        return self

    def commit_to_hugegraph(self):
        self.operators.append(CommitToKg())
        return self

    def build_vertex_id_semantic_index(self):
        self.operators.append(BuildSemanticIndex(self.embedding))
        return self

    def build_vector_index(self):
        self.operators.append(BuildVectorIndex(self.embedding))
        return self

    def print_result(self):
        self.operators.append(PrintResult())
        return self

    def run(self) -> Dict[str, Any]:
        context = None
        for operator in self.operators:
            log.debug("Running operator: %s", operator.__class__.__name__)
            start = time.time()
            context = operator.run(context)
            log.debug("Operator %s finished in %s seconds", operator.__class__.__name__,
                      time.time() - start)
            log.debug("Context:\n%s", context)
        return context
