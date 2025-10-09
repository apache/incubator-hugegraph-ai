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


import re
import time
from typing import Set, Dict, Any, Optional

from hugegraph_llm.config import prompt
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper
from hugegraph_llm.utils.log import log

KEYWORDS_EXTRACT_TPL = prompt.keywords_extract_prompt


class KeywordExtract:
    def __init__(
        self,
        text: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        max_keywords: int = 5,
        extract_template: Optional[str] = None,
        language: str = "english",
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._query is None:
            self._query = context.get("query")
            assert self._query is not None, "No query for keywords extraction."
        else:
            context["query"] = self._query

        if self._llm is None:
            self._llm = LLMs().get_extract_llm()
            assert isinstance(self._llm, BaseLLM), "Invalid LLM Object."

        self._language = context.get("language", self._language).lower()
        self._max_keywords = context.get("max_keywords", self._max_keywords)

        prompt_run = f"{self._extract_template.format(question=self._query, max_keywords=self._max_keywords)}"
        start_time = time.perf_counter()
        response = self._llm.generate(prompt=prompt_run)
        end_time = time.perf_counter()
        log.debug("Keyword extraction time: %.2f seconds", end_time - start_time)

        keywords = self._extract_keywords_from_response(
            response=response, lowercase=False, start_token="KEYWORDS:"
        )
        keywords = {k.replace("'", "") for k in keywords}
        context["keywords"] = list(keywords)
        log.info("User Query: %s\nKeywords: %s", self._query, context["keywords"])

        # extracting keywords & expanding synonyms increase the call count by 1
        context["call_count"] = context.get("call_count", 0) + 1
        return context

    def _extract_keywords_from_response(
        self,
        response: str,
        lowercase: bool = True,
        start_token: str = "",
    ) -> Set[str]:
        keywords = []
        # use re.escape(start_token) if start_token contains special chars like */&/^ etc.
        matches = re.findall(rf"{start_token}[^\n]+\n?", response)

        for match in matches:
            match = match[len(start_token) :].strip()
            keywords.extend(
                k.lower() if lowercase else k
                for k in re.split(r"[,，]+", match)
                if len(k.strip()) > 1
            )

        # if the keyword consists of multiple words, split into sub-words (removing stopwords)
        results = set(keywords)
        for token in keywords:
            sub_tokens = re.findall(r"\w+", token)
            if len(sub_tokens) > 1:
                results.update(
                    w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language)
                )
        return results
