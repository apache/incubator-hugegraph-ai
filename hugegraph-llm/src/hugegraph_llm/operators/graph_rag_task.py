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
from typing import Dict, Any, Optional, List

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.common_op.merge_dedup_rerank import MergeDedupRerank
from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery
from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery
from hugegraph_llm.operators.llm_op.answer_synthesize import AnswerSynthesize
from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract
from hugegraph_llm.utils.log import log


class GraphRAG:
    def __init__(self, llm: Optional[BaseLLM] = None, embedding: Optional[BaseEmbedding] = None):
        self._llm = llm or LLMs().get_llm()
        self._embedding = embedding or Embeddings().get_embedding()
        self._operators: List[Any] = []

    def extract_keyword(
        self,
        text: Optional[str] = None,
        max_keywords: int = 5,
        language: str = "english",
        extract_template: Optional[str] = None,
        expand_template: Optional[str] = None,
    ):
        self._operators.append(
            KeywordExtract(
                text=text,
                max_keywords=max_keywords,
                language=language,
                extract_template=extract_template,
                expand_template=expand_template,
            )
        )
        return self

    def match_keyword_to_id(self, topk_per_keyword: int = 1):
        self._operators.append(
            SemanticIdQuery(
                embedding=self._embedding,
                topk_per_keyword=topk_per_keyword
            )
        )
        return self

    def query_graph_for_rag(
        self,
        max_deep: int = 2,
        max_items: int = 30,
        prop_to_match: Optional[str] = None,
    ):
        self._operators.append(
            GraphRAGQuery(
                max_deep=max_deep,
                max_items=max_items,
                prop_to_match=prop_to_match,
            )
        )
        return self

    def query_vector_index_for_rag(
            self,
            max_items: int = 3
    ):
        self._operators.append(
            VectorIndexQuery(
                embedding=self._embedding,
                topk=max_items,
            )
        )
        return self

    def merge_dedup_rerank(self):
        self._operators.append(
            MergeDedupRerank(
                embedding=self._embedding,
            )
        )
        return self

    def synthesize_answer(
        self,
        raw_answer: bool = False,
        vector_only_answer: bool = True,
        graph_only_answer: bool = False,
        graph_vector_answer: bool = False,
        prompt_template: Optional[str] = None,
    ):
        self._operators.append(
            AnswerSynthesize(
                raw_answer = raw_answer,
                vector_only_answer = vector_only_answer,
                graph_only_answer = graph_only_answer,
                graph_vector_answer = graph_vector_answer,
                prompt_template=prompt_template,
            )
        )
        return self

    def print_result(self):
        self._operators.append(PrintResult())
        return self

    def run(self, **kwargs) -> Dict[str, Any]:
        if len(self._operators) == 0:
            self.extract_keyword().query_graph_for_rag().synthesize_answer()

        context = kwargs
        context["llm"] = self._llm
        for operator in self._operators:
            log.debug("Running operator: %s", operator.__class__.__name__)
            start = time.time()
            context = operator.run(context)
            log.debug("Operator %s finished in %s seconds", operator.__class__.__name__,
                      time.time() - start)
            log.debug("Context:\n%s", context)
        return context
