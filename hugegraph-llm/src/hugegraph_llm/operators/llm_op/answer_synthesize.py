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

# pylint: disable=W0621

import asyncio
from typing import Any, Dict, List, Optional

from hugegraph_llm.config import prompt
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log

DEFAULT_ANSWER_TEMPLATE = prompt.answer_prompt


def _get_vector_result_str(vector_result: List[str]) -> str:
    if vector_result:
        return "Phrases related to the query:\n" + "\n".join(
            f"{i + 1}. {res}" for i, res in enumerate(vector_result)
        )
    no_vector_data_msg = "No (vector)phrase related to the query."
    log.warning(no_vector_data_msg)
    return no_vector_data_msg

def _get_graph_result_str(graph_context_head: str, graph_result: List[str]) -> str:
    if graph_result:
        return graph_context_head + "\n".join(
            f"{i + 1}. {res}" for i, res in enumerate(graph_result)
        )
    no_graph_data_msg = "No related graph data found for current query."
    log.warning(no_graph_data_msg)
    return no_graph_data_msg

class AnswerSynthesize:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = None,
        question: Optional[str] = None,
        context_body: Optional[str] = None,
        context_head: Optional[str] = None,
        context_tail: Optional[str] = None,
    ):
        self._llm = llm
        self._prompt_template = prompt_template or DEFAULT_ANSWER_TEMPLATE
        self._question = question
        self._context_body = context_body
        self._context_head = context_head
        self._context_tail = context_tail

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._llm is None:
            self._llm = context.get("llm") or LLMs().get_llm()
        if context.get("llm") is None:
            context["llm"] = self._llm

        if self._question is None:
            self._question = context.get("query") or None
        assert self._question is not None, "No question for synthesizing."

        context_head_str = (
            context.get("synthesize_context_head") or self._context_head or ""
        )
        context_tail_str = (
            context.get("synthesize_context_tail") or self._context_tail or ""
        )

        if self._context_body is not None:
            context_str = (
                f"{context_head_str}\n"
                f"{self._context_body}\n"
                f"{context_tail_str}".strip("\n")
            )

            final_prompt = self._prompt_template.format(
                context_str=context_str, query_str=self._question
            )
            response = self._llm.generate(prompt=final_prompt)
            return {"answer": response}

        context = asyncio.run(
            self.async_generate(
                context,
                context_head_str,
                context_tail_str
            )
        )
        return context

    async def async_generate(
        self,
        context: Dict[str, Any],
        context_head_str: str,
        context_tail_str: str
    ):
        # pylint: disable=R0912 (too-many-branches)
        verbose = context.get("verbose") or False
        # TODO: replace task_cache with a better name
        task_cache = {}

        raw_answer = context.get("raw_answer", False)
        vector_only_answer = context.get("vector_only_answer", False)
        graph_only_answer = context.get("graph_only_answer", False)
        graph_vector_answer = context.get("graph_vector_answer", False)

        if raw_answer:
            final_prompt = self._question
            task_cache["raw_task"] = asyncio.create_task(
                self._llm.agenerate(prompt=final_prompt)
            )
        if vector_only_answer:
            vector_result = context.get("vector_result")
            vector_result_context = _get_vector_result_str(vector_result)

            context_str = (
                f"{context_head_str}\n"
                f"{vector_result_context}\n"
                f"{context_tail_str}".strip("\n")
            )
            context["vector_contexts"] = vector_result

            final_prompt = self._prompt_template.format(
                context_str=context_str, query_str=self._question
            )
            task_cache["vector_only_task"] = asyncio.create_task(
                self._llm.agenerate(prompt=final_prompt)
            )
        if graph_only_answer:

            graph_result = context.get("graph_result")
            graph_context_head = context.get(
                "graph_context_head", "Knowledge from graphdb for the query:\n"
            )
            graph_result_context = _get_graph_result_str(graph_context_head, graph_result)
            
            context_str = (
                f"{context_head_str}\n"
                f"{graph_result_context}\n"
                f"{context_tail_str}".strip("\n")
            )
            context["graph_contexts"] = graph_result

            final_prompt = self._prompt_template.format(
                context_str=context_str, query_str=self._question
            )
            task_cache["graph_only_task"] = asyncio.create_task(
                self._llm.agenerate(prompt=final_prompt)
            )
        if graph_vector_answer:
            vector_result = context.get("vector_result")
            vector_rerank_length = context.get("vector_rerank_length")
            vector_result_context = _get_vector_result_str(vector_result[:vector_rerank_length])
            
            graph_result = context.get("graph_result")
            graph_rerank_length = context.get("graph_rerank_length")
            graph_context_head = context.get(
                "graph_context_head", "Knowledge from graphdb for the query:\n"
            )
            graph_result_context = _get_graph_result_str(graph_context_head, graph_result[:graph_rerank_length])
            context_body_str = f"{vector_result_context}\n{graph_result_context}"

            context["graph_vector_contexts"] = vector_result[:vector_rerank_length] + graph_result[:graph_rerank_length]

            if context.get("graph_ratio", 0.5) < 0.5:
                context_body_str = f"{graph_result_context}\n{vector_result_context}"
            context_str = (
                f"{context_head_str}\n"
                f"{context_body_str}\n"
                f"{context_tail_str}".strip("\n")
            )

            final_prompt = self._prompt_template.format(
                context_str=context_str, query_str=self._question
            )
            
            task_cache["graph_vector_task"] = asyncio.create_task(
                self._llm.agenerate(prompt=final_prompt)
            )
        # TODO: use log.debug instead of print
        if task_cache.get("raw_task"):
            response = await task_cache["raw_task"]
            context["raw_answer_result"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("vector_only_task"):
            response = await task_cache["vector_only_task"]
            context["vector_only_answer_result"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("graph_only_task"):
            response = await task_cache["graph_only_task"]
            context["graph_only_answer_result"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")
        if task_cache.get("graph_vector_task"):
            response = await task_cache["graph_vector_task"]
            context["graph_vector_answer_result"] = response
            if verbose:
                print(f"\033[91mANSWER: {response}\033[0m")

        ops = sum(
            [raw_answer, vector_only_answer, graph_only_answer, graph_vector_answer]
        )
        context["call_count"] = context.get("call_count", 0) + ops
        return context
