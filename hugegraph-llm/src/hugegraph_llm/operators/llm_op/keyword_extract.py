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
from typing import Any, Dict, Optional, Set

from hugegraph_llm.config import prompt
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper
from hugegraph_llm.operators.document_op.textrank_word_extract import MultiLingualTextRank
from hugegraph_llm.utils.log import log

KEYWORDS_EXTRACT_TPL = prompt.keywords_extract_prompt


class KeywordExtract:
    def __init__(
        self,
        text: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        max_keywords: int = 5,
        extract_template: Optional[str] = None,
        extract_method: str = "Hybrid",  # "Hybrid", "LLM", "TextRank"
        mask_words: str = "",
    ):
        self._llm = llm
        self._query = text
        self._language = "english"
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL
        self._extract_method = extract_method
        self._textrank_model = MultiLingualTextRank(keyword_num=max_keywords, mask_words=mask_words)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._query is None:
            self._query = context.get("query")
            assert self._query is not None, "No query for keywords extraction."
        else:
            context["query"] = self._query

        if self._llm is None:
            self._llm = LLMs().get_extract_llm()
            assert isinstance(self._llm, BaseLLM), "Invalid LLM Object."

        lang = context.get("language", self._language).lower()
        if lang in ("en", "zh"):
            lang = "english" if lang == "en" else "chinese"
        self._language = lang
        self._max_keywords = context.get("max_keywords", self._max_keywords)

        if self._extract_method == "TextRank":
            # 使用 TextRank 提取关键词
            keywords = self._extract_with_textrank()
        elif self._extract_method == "LLM":
            # 使用 LLM 提取关键词
            keywords = self._extract_with_llm()
        elif self._extract_method == "Hybrid":
            keywords = self._extract_with_hybrid()
        else:
            raise ValueError(f"Invalid extract_method: {self._extract_method}")

        keywords = {k.replace("'", "") for k in keywords}
        context["keywords"] = list(keywords)[:self._max_keywords]
        log.info("User Query: %s\nKeywords: %s", self._query, context["keywords"])

        # extracting keywords & expanding synonyms increase the call count by 1
        context["call_count"] = context.get("call_count", 0) + 1
        return context

    def _extract_with_llm(self) -> Set[str]:
        prompt_run = f"{self._extract_template.format(question=self._query, max_keywords=self._max_keywords)}"
        start_time = time.perf_counter()
        response = self._llm.generate(prompt=prompt_run)
        end_time = time.perf_counter()
        log.debug("LLM Keyword extraction time: %.2f seconds", end_time - start_time)
        keywords = self._extract_keywords_from_response(
            response=response, lowercase=False, start_token="KEYWORDS:"
        )
        return keywords

    def _extract_with_textrank(self) -> Set[str]:
        """ TextRank 提取模式 """
        start_time = time.perf_counter()
        try:
            keywords = self._textrank_model.extract_keywords(self._query)
        except FileNotFoundError as e:
            log.error("TextRank resource file not found: %s", e)
            keywords = []
        except (TypeError, ValueError) as e:
            log.error("TextRank parameter error: %s", e)
            keywords = []
        except MemoryError as e:
            log.error("TextRank memory error (text too large?): %s", e)
            keywords = []
        log.debug("TextRank Keyword extraction time: %.2f seconds",
                  time.perf_counter() - start_time)
        return set(filter(None, keywords))

    def _extract_with_hybrid(self) -> Set[str]:
        """
        Hybrid mode with a "Full-Match Intersection" strategy.
        The priority order is:
        1. Intersection keywords (LLM phrases fully matched by TextRank parts).
        2. Remaining LLM keywords.
        3. Remaining TextRank keywords.
        """
        llm_keywords = self._extract_with_llm()
        textrank_keywords = self._extract_with_textrank()
        tr_lower = {t.lower() for t in textrank_keywords}
        log.debug("LLM keywords: %s, TextRank keywords: %s", llm_keywords, textrank_keywords)

        intersection_keywords = list()
        used_tr_keywords = list()

        for lk in llm_keywords:
            word = lk.lower()
            parts = re.split(r'\s+', word)
            if len(parts) > 1:
                # Multi-word phrase: check if all parts are in TextRank
                if all(part in tr_lower for part in parts):
                    intersection_keywords.append(word)
                    used_tr_keywords.append(part for part in parts)
            else:
                # Single-word keyword: check for direct existence
                if word in tr_lower:
                    intersection_keywords.append(word)
                    used_tr_keywords.append(word)

        remaining_llm = [lk for lk in llm_keywords if lk.lower() not in intersection_keywords]
        remaining_textrank = [trk for trk in textrank_keywords if trk.lower() not in used_tr_keywords]

        ordered_keywords = intersection_keywords + remaining_llm + remaining_textrank
        ordered_keywords = ordered_keywords[:self._max_keywords]
        return set(ordered_keywords)

    def _extract_keywords_from_response(
        self,
        response: str,
        lowercase: bool = True,
        start_token: str = "",
    ) -> Set[str]:
        keywords = []
        # use re.escape(start_token) if start_token contains special chars like */&/^ etc.
        matches = re.findall(rf'{start_token}[^\n]+\n?', response)

        for match in matches:
            match = match[len(start_token):].strip()
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
                    w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language))
        return results
