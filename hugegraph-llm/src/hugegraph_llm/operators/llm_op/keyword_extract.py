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
from typing import Any, Dict, Optional, List

import jieba.posseg as pseg

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

        if self._extract_method == "LLM":
            # 使用 LLM 提取关键词
            keywords = self._extract_with_llm()
        elif self._extract_method == "TextRank":
            # 使用 TextRank 提取关键词
            ranks = self._extract_with_textrank()
            keywords = [] if not ranks else sorted(ranks, key=ranks.get, reverse=True)
        elif self._extract_method == "Hybrid":
            # 使用 混合方法 提取关键词
            keywords = self._extract_with_hybrid()
        else:
            raise ValueError(f"Invalid extract_method: {self._extract_method}")

        keywords = [k.replace("'", "") for k in keywords]
        context["keywords"] = keywords[:self._max_keywords]
        log.info("User Query: %s\nKeywords: %s", self._query, context["keywords"])

        # extracting keywords & expanding synonyms increase the call count by 1
        context["call_count"] = context.get("call_count", 0) + 1
        return context

    def _extract_with_llm(self) -> List[str]:
        prompt_run = f"{self._extract_template.format(question=self._query, max_keywords=self._max_keywords)}"
        start_time = time.perf_counter()
        response = self._llm.generate(prompt=prompt_run)
        end_time = time.perf_counter()
        log.debug("LLM Keyword extraction time: %.2f seconds", end_time - start_time)
        keywords = self._extract_keywords_from_response(
            response=response, lowercase=False, start_token="KEYWORDS:"
        )
        return keywords

    def _extract_with_textrank(self) -> Dict:
        """ TextRank 提取模式 """
        start_time = time.perf_counter()
        ranks = {}
        try:
            ranks = self._textrank_model.extract_keywords(self._query)
        except (TypeError, ValueError) as e:
            log.error("TextRank parameter error: %s", e)
        except MemoryError as e:
            log.critical("TextRank memory error (text too large?): %s", e)
        end_time = time.perf_counter()
        log.debug("TextRank Keyword extraction time: %.2f seconds",
                  end_time - start_time)
        return ranks

    def _extract_with_hybrid(self) -> List[str]:
        """
        Hybrid mode extraction
        Source Scores: Intersection: 1.0, only from llm: 0.8, only from textrank: 0.5
        TextRank Scores: scores from TextRank, for llm long keywords, its scores is sum of its token's scores
        len Scores:
        """
        start_time = time.perf_counter()
        llm_keywords = self._extract_with_llm()[:self._max_keywords]
        ranks = self._extract_with_textrank()
        scores_w = [0.5, 0.3, 0.2]

        scores_map = {}
        token_map = {}
        word_set = set()
        lower_ranks = {k.lower(): v for k, v in ranks.items()}

        # 1. LLM parts
        for lk in llm_keywords:
            parts = re.split(r'\s+', lk.lower())
            scores_map[lk] = 0
            token_map[lk] = len(parts)
            word_set.add(lk.lower())
            if all(part in lower_ranks for part in parts):
                scores_map[lk] += 1 * scores_w[0]
            else:
                scores_map[lk] += 0.8 * scores_w[0]
            for part in parts:
                if part in lower_ranks:
                    scores_map[lk] += lower_ranks[part] * scores_w[1]

        # 2. TextRank parts
        for word, score in ranks.items():
            if word.lower() not in word_set:
                token_map[word] = 1
                scores_map[word] = 0.5 * scores_w[0] + score * scores_w[1]

        # 3. calculate scores
        if len(scores_map) > self._max_keywords:
            max_token_len = max(token_map.values())
            scores_map = {k: v + token_map[k] * scores_w[2] / max_token_len for k, v in scores_map.items()}
            ordered_keywords = sorted(scores_map, key=scores_map.get, reverse=True)
        else:
            ordered_keywords = list(scores_map.keys())

        end_time = time.perf_counter()
        log.debug("Hybrid Keyword extraction time: %.2f seconds", end_time - start_time)
        return ordered_keywords

    def _extract_keywords_from_response(
        self,
        response: str,
        lowercase: bool = True,
        start_token: str = "",
    ) -> List[str]:

        lower_keywords: set[str] = set()
        results = []

        # use re.escape(start_token) if start_token contains special chars like */&/^ etc.
        matches = re.findall(rf'{start_token}[^\n]+\n?', response)

        for match in matches:
            match = match[len(start_token):].strip()
            if match not in lower_keywords:
                for k in re.split(r"[,，]+", match):
                    word = k.strip()
                    if len(word) > 1 and word not in lower_keywords:
                        results.append(word.lower() if lowercase else word)
                        lower_keywords.add(word.lower())

        sub_tokens = []
        # if the keyword consists of multiple words, split into sub-words (removing stopwords)
        for token in list(results):
            if re.compile('[\u4e00-\u9fa5]').search(token) is None:
                sub_tokens = re.findall(r"\w+", token)
            else:
                sub_tokens = [w for (w, _flag) in pseg.cut(token)]
            if len(sub_tokens) > 1:
                for w in sub_tokens:
                    lw = w.lower()
                    if lw not in NLTKHelper().stopwords(lang=self._language) and lw not in lower_keywords:
                        sub_tokens.append(lw if lowercase else w)
                        lower_keywords.add(lw)
        if sub_tokens:
            results.extend(sub_tokens)

        return results
