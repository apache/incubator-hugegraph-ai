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


from typing import Dict, Any, Optional, List, Literal

from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.merge_dedup_rerank import MergeDedupRerank
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.document_op.word_extract import WordExtract
from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery
from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery
from hugegraph_llm.operators.llm_op.answer_synthesize import AnswerSynthesize
from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract
from hugegraph_llm.utils.decorators import log_time, log_operator_time, log_llm_qps


class RAGPipeline:
    """
    RAGPipeline is a (core)class that encapsulates a series of operations for extracting information from text,
    querying graph databases and vector indices, merging and re-ranking results, and generating answers.
    """

    def __init__(self, llm: Optional[BaseLLM] = None, embedding: Optional[BaseEmbedding] = None):
        """
        Initialize the RAGPipeline with optional LLM and embedding models.

        :param llm: Optional LLM model to use.
        :param embedding: Optional embedding model to use.
        """
        self._llm = llm or LLMs().get_llm()
        self._embedding = embedding or Embeddings().get_embedding()
        self._operators: List[Any] = []

    def extract_word(
            self,
            text: Optional[str] = None,
            language: str = "english",
    ):
        """
        Add a word extraction operator to the pipeline.

        :param text: Text to extract words from.
        :param language: Language of the text.
        :return: Self-instance for chaining.
        """
        self._operators.append(
            WordExtract(
                text=text,
                language=language,
            )
        )
        return self

    def extract_keyword(
            self,
            text: Optional[str] = None,
            max_keywords: int = 5,
            language: str = "english",
            extract_template: Optional[str] = None,
            expand_template: Optional[str] = None,
    ):
        """
        Add a keyword extraction operator to the pipeline.

        :param text: Text to extract keywords from.
        :param max_keywords: Maximum number of keywords to extract.
        :param language: Language of the text.
        :param extract_template: Template for keyword extraction.
        :param expand_template: Template for keyword expansion.
        :return: Self-instance for chaining.
        """
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

    def match_keyword_to_id(
            self,
            by: Literal["query", "keywords"] = "keywords",
            topk_per_keyword: int = 1,
            topk_per_query: int = 10
    ):
        """
        Add a semantic ID query operator to the pipeline.

        :param topk_per_keyword: Top K results per keyword.
        :return: Self-instance for chaining.
        """
        self._operators.append(
            SemanticIdQuery(
                embedding=self._embedding,
                by=by,
                topk_per_keyword=topk_per_keyword,
                topk_per_query=topk_per_query
            )
        )
        return self

    def query_graph_for_rag(
            self,
            max_deep: int = 2,
            max_items: int = 30,
            prop_to_match: Optional[str] = None,
    ):
        """
        Add a graph RAG query operator to the pipeline.

        :param max_deep: Maximum depth for the graph query.
        :param max_items: Maximum number of items to retrieve.
        :param prop_to_match: Property to match in the graph.
        :return: Self-instance for chaining.
        """
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
        """
        Add a vector index query operator to the pipeline.

        :param max_items: Maximum number of items to retrieve.
        :return: Self-instance for chaining.
        """
        self._operators.append(
            VectorIndexQuery(
                embedding=self._embedding,
                topk=max_items,
            )
        )
        return self

    def merge_dedup_rerank(self):
        """
        Add a merge, deduplication, and rerank operator to the pipeline.

        :return: Self-instance for chaining.
        """
        self._operators.append(MergeDedupRerank(embedding=self._embedding))
        return self

    def synthesize_answer(
            self,
            raw_answer: bool = False,
            vector_only_answer: bool = True,
            graph_only_answer: bool = False,
            graph_vector_answer: bool = False,
            answer_prompt: Optional[str] = None,
    ):
        """
        Add an answer synthesis operator to the pipeline.

        :param raw_answer: Whether to return raw answers.
        :param vector_only_answer: Whether to return vector-only answers.
        :param graph_only_answer: Whether to return graph-only answers.
        :param graph_vector_answer: Whether to return graph-vector combined answers.
        :param answer_prompt: Template for the answer synthesis prompt.
        :return: Self-instance for chaining.
        """
        self._operators.append(
            AnswerSynthesize(
                raw_answer=raw_answer,
                vector_only_answer=vector_only_answer,
                graph_only_answer=graph_only_answer,
                graph_vector_answer=graph_vector_answer,
                prompt_template=answer_prompt,
            )
        )
        return self

    def print_result(self):
        """
        Add a print result operator to the pipeline.

        :return: Self-instance for chaining.
        """
        self._operators.append(PrintResult())
        return self

    @log_time("total time")
    @log_llm_qps
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all operators in the pipeline in sequence.

        :param kwargs: Additional context to pass to operators.
        :return: Final context after all operators have been executed.
        """
        if len(self._operators) == 0:
            self.extract_keyword().query_graph_for_rag().synthesize_answer()

        context = kwargs
        context["llm"] = self._llm

        for operator in self._operators:
            context = self._run_operator(operator, context)
        return context

    @log_operator_time
    def _run_operator(self, operator, context):
        return operator.run(context)
