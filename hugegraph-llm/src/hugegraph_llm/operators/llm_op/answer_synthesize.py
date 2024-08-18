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
from typing import Any, Dict, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs

# TODO: we need enhance the template to answer the question
DEFAULT_ANSWER_SYNTHESIZE_TEMPLATE_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You need to refer to the context based on the following priority:\n"
    "1. Graph recall > vector recall\n"
    "2. Exact recall > Fuzzy recall\n"
    "3. Independent vertex > 1-depth neighbor> 2-depth neighbors\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


class AnswerSynthesize:
    def __init__(
            self,
            llm: Optional[BaseLLM] = None,
            prompt_template: Optional[str] = None,
            question: Optional[str] = None,
            context_body: Optional[str] = None,
            context_head: Optional[str] = None,
            context_tail: Optional[str] = None,
            raw_answer: bool = False,
            vector_only_answer: bool = True,
            graph_only_answer: bool = False,
            graph_vector_answer: bool = False,
    ):
        self._llm = llm
        self._prompt_template = prompt_template or DEFAULT_ANSWER_SYNTHESIZE_TEMPLATE_TMPL
        self._question = question
        self._context_body = context_body
        self._context_head = context_head
        self._context_tail = context_tail
        self._raw_answer = raw_answer
        self._vector_only_answer = vector_only_answer
        self._graph_only_answer = graph_only_answer
        self._graph_vector_answer = graph_vector_answer

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._llm is None:
            self._llm = context.get("llm") or LLMs().get_llm()
        if context.get("llm") is None:
            context["llm"] = self._llm

        if self._question is None:
            self._question = context.get("query") or None
        assert self._question is not None, "No question for synthesizing."

        context_head_str = context.get("synthesize_context_head") or self._context_head or ""
        context_tail_str = context.get("synthesize_context_tail") or self._context_tail or ""

        if self._context_body is not None:
            context_str = (f"{context_head_str}\n"
                           f"{self._context_body}\n"
                           f"{context_tail_str}".strip("\n"))

            prompt = self._prompt_template.format(
                context_str=context_str,
                query_str=self._question,
            )
            response = self._llm.generate(prompt=prompt)
            return {"answer": response}

        vector_result = context.get("vector_result", [])
        if len(vector_result) == 0:
            vector_result_context = "There are no paragraphs related to the query."
        else:
            vector_result_context = ("The following are paragraphs related to the query:\n"
                                     + "\n".join([f"{i + 1}. {res}"
                                                  for i, res in enumerate(vector_result)]))
        graph_result = context.get("graph_result", [])
        if len(graph_result) == 0:
            graph_result_context = "There are no knowledge from HugeGraph related to the query."
        else:
            graph_result_context = (
                    "The following are knowledge from HugeGraph related to the query:\n"
                    + "\n".join([f"{i + 1}. {res}"
                                 for i, res in enumerate(graph_result)]))
        context = asyncio.run(self.async_generate(context, context_head_str, context_tail_str,
                                                  vector_result_context, graph_result_context))

        return context

    async def async_generate(self, context: Dict[str, Any], context_head_str: str,
                             context_tail_str: str, vector_result_context: str,
                             graph_result_context: str):
        verbose = context.get("verbose") or False
        task_cache = {}
        if self._raw_answer:
            prompt = self._question
            task_cache["raw_task"] = asyncio.create_task(self._llm.agenerate(prompt=prompt))
        if self._vector_only_answer:
            context_str = (f"{context_head_str}\n"
                           f"{vector_result_context}\n"
                           f"{context_tail_str}".strip("\n"))

            prompt = self._prompt_template.format(
                context_str=context_str,
                query_str=self._question,
            )
            task_cache["vector_only_task"] = asyncio.create_task(self._llm.agenerate(prompt=prompt))
        if self._graph_only_answer:
            context_str = (f"{context_head_str}\n"
                           f"{graph_result_context}\n"
                           f"{context_tail_str}".strip("\n"))

            prompt = self._prompt_template.format(
                context_str=context_str,
                query_str=self._question,
            )
            task_cache["graph_only_task"] = asyncio.create_task(self._llm.agenerate(prompt=prompt))
        if self._graph_vector_answer:
            context_body_str = f"{vector_result_context}\n{graph_result_context}"
            context_str = (f"{context_head_str}\n"
                           f"{context_body_str}\n"
                           f"{context_tail_str}".strip("\n"))

            prompt = self._prompt_template.format(
                context_str=context_str,
                query_str=self._question,
            )
            task_cache["graph_vector_task"] = asyncio.create_task(
                self._llm.agenerate(prompt=prompt)
            )
        if task_cache.get("raw_task"):
            response = await task_cache["raw_task"]
            context["raw_answer"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("vector_only_task"):
            response = await task_cache["vector_only_task"]
            context["vector_only_answer"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("graph_only_task"):
            response = await task_cache["graph_only_task"]
            context["graph_only_answer"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("graph_vector_task"):
            response = await task_cache["graph_vector_task"]
            context["graph_vector_answer"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        return context
