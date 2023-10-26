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


from typing import Dict, Any, Optional, List

from hugegraph_llm.llms.base import BaseLLM
from hugegraph_llm.llms.openai_llm import OpenAIChat
from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
from hugegraph_llm.operators.llm_op.answer_synthesize import AnswerSynthesize
from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract
from pyhugegraph.client import PyHugeClient


class GraphRAG:
    def __init__(self, llm: Optional[BaseLLM] = None):
        self._llm = llm or OpenAIChat()
        self._operators: List[Any] = []

    def extract_keyword(
            self,
            text: Optional[str] = None,
            max_keywords: int = 5,
            language: str = 'english',
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

    def query_graph_for_rag(
            self,
            graph_client: Optional[PyHugeClient] = None,
            max_deep: int = 2,
            max_items: int = 30,
            prop_to_match: Optional[str] = None,
    ):
        self._operators.append(
            GraphRAGQuery(
                client=graph_client,
                max_deep=max_deep,
                max_items=max_items,
                prop_to_match=prop_to_match,
            )
        )
        return self

    def synthesize_answer(
            self,
            prompt_template: Optional[str] = None,
    ):
        self._operators.append(
            AnswerSynthesize(
                prompt_template=prompt_template,
            )
        )
        return self

    def run(self, **kwargs) -> Dict[str, Any]:
        if len(self._operators) == 0:
            self.extract_keyword().query_graph_for_rag().synthesize_answer()

        context = kwargs
        context["llm"] = self._llm
        for op in self._operators:
            context = op.run(context=context)
        return context
