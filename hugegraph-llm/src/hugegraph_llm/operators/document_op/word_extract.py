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
from typing import Dict, Any, Optional, List

import jieba

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper


class WordExtract:
    def __init__(
        self,
        text: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        language: str = "english",
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._query is None:
            self._query = context.get("query")
            assert self._query is not None, "No query for keywords extraction."
        else:
            context["query"] = self._query

        if self._llm is None:
            self._llm = context.get("llm") or LLMs().get_llm()
            assert isinstance(self._llm, BaseLLM), "Invalid LLM Object."
        if context.get("llm") is None:
            context["llm"] = self._llm

        if isinstance(context.get("language"), str):
            self._language = context["language"].lower()
        else:
            context["language"] = self._language

        keywords = jieba.lcut(self._query)
        keywords = self._filter_keywords(keywords, lowercase=False)

        context["keywords"] = keywords

        verbose = context.get("verbose") or False
        if verbose:
            from hugegraph_llm.utils.log import log
            log.info("KEYWORDS: %s", context['keywords'])

        return context

    def _filter_keywords(
        self,
        keywords: List[str],
        lowercase: bool = True,
    ) -> List[str]:
        if lowercase:
            keywords = [w.lower() for w in keywords]

        # if the keyword consists of multiple words, split into sub-words
        # (removing stopwords)
        results = set()
        for token in keywords:
            results.add(token)
            sub_tokens = re.findall(r"\w+", token)
            if len(sub_tokens) > 1:
                results.update(
                    {w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language)}
                )

        return list(results)
