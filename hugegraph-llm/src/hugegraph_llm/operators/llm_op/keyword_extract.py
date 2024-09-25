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
from typing import Set, Dict, Any, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper

KEYWORDS_EXTRACT_TPL = """extract {max_keywords} keywords from the text:
{question}
Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'
"""

KEYWORDS_EXPAND_TPL = """Generate synonyms or possible form of keywords up to {max_keywords} in total,
considering possible cases of capitalization, pluralization, common expressions, etc.
Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <keywords>'
Note, result should be in one-line with only one 'SYNONYMS: ' prefix
----
KEYWORDS: {question}
----"""


class KeywordExtract:
    def __init__(
        self,
        text: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        max_keywords: int = 5,
        extract_template: Optional[str] = None,
        expand_template: Optional[str] = None,
        language: str = "english",
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL
        self._expand_template = expand_template or KEYWORDS_EXPAND_TPL

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._query is None:
            self._query = context.get("query")
            assert self._query is not None, "No query for keywords extraction."
        else:
            context["query"] = self._query

        if self._llm is None:
            self._llm = LLMs().get_llm()
            assert isinstance(self._llm, BaseLLM), "Invalid LLM Object."

        if isinstance(context.get("language"), str):
            self._language = context["language"].lower()
        else:
            context["language"] = self._language

        if isinstance(context.get("max_keywords"), int):
            self._max_keywords = context["max_keywords"]

        prompt = self._extract_template.format(
            question=self._query,
            max_keywords=self._max_keywords,
        )
        response = self._llm.generate(prompt=prompt)

        keywords = self._extract_keywords_from_response(
            response=response, lowercase=False, start_token="KEYWORDS:"
        )
        keywords.union(self._expand_synonyms(keywords=keywords))
        context["keywords"] = list(keywords)

        verbose = context.get("verbose") or False
        if verbose:
            from hugegraph_llm.utils.log import log
            log.info("KEYWORDS: %s", context['keywords'])

        # extracting keywords & expanding synonyms increase the call count by 2
        context["call_count"] = context.get("call_count", 0) + 2
        return context

    def _expand_synonyms(self, keywords: Set[str]) -> Set[str]:
        prompt = self._expand_template.format(
            question=str(keywords),
            max_keywords=self._max_keywords,
        )
        response = self._llm.generate(prompt=prompt)
        keywords = self._extract_keywords_from_response(
            response=response, lowercase=False, start_token="SYNONYMS:"
        )
        return keywords

    def _extract_keywords_from_response(
        self,
        response: str,
        lowercase: bool = True,
        start_token: str = "",
    ) -> Set[str]:
        keywords = []
        matches = re.findall(rf'{start_token}[^\n]+\n?', response)
        for match in matches:
            match = match[len(start_token):]
            for k in re.split(r"[,，]+", match):
                k = k.strip()
                if len(k) > 0:
                    if lowercase:
                        keywords.append(k.lower())
                    else:
                        keywords.append(k)

        # if the keyword consists of multiple words, split into sub-words (removing stopwords)
        results = set()
        for token in keywords:
            results.add(token)
            sub_tokens = re.findall(r"\w+", token)
            if len(sub_tokens) > 1:
                results.update({w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language)})
        return results
