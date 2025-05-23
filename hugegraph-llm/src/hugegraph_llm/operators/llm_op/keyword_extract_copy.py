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
from gensim.summarization import keywords as textrank_keywords
import jieba

import sys
sys.path.append('/mnt/WD4T/workspace/hs/incubator-hugegraph-ai/hugegraph-llm/src')

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.config import prompt
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
            use_textrank: bool = False,  # 新增TextRank开关
            textrank_kwargs: Optional[Dict] = None,  # TextRank参数
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL
        self._use_textrank = use_textrank  # 新增TextRank开关
        self._textrank_config = {
            "ratio": 0.2,  # 提取前20%的关键词
            "scores": False,  # 不返回关键词的分数
            **(textrank_kwargs or {})
        }  # TextRank参数

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

        if self._use_textrank:
            # 使用TextRank提取关键词
            keywords = self._extract_with_textrank()
        else:
            # 使用LLM提取关键词
            keywords = self._extract_with_llm()
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
        """ TextRank提取模式 """
        start_time = time.perf_counter()
        # 多语言预处理
        if self._language.startswith("zh"):
            words = jieba.lcut(self._query)  # 中文分词
            processed_text = " ".join(words)
        else:
            processed_text = self._query  # 英文保持原始文本

        try:
            # 使用Gensim的TextRank实现
            keywords = textrank_keywords(
                processed_text, 
                words=self._max_keywords,
                **self._textrank_config
            ).split("\n")
        except Exception as e:
            log.error(f"TextRank提取失败: {str(e)}")
            keywords = []
        log.debug(f"TextRank提取耗时: {time.perf_counter()-start_time:.2f}s")
    
        return set(filter(None, keywords))

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
                results.update(w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language))
        return results

def test_textrank_english():
    """测试英文TextRank提取"""
    extractor = KeywordExtract(
        text="Natural language processing (NLP) is a subfield of AI focused on computer-human interaction. It enables machines to understand human language.",
        use_textrank=True,
        max_keywords=3,
        language="english"
    )
    result = extractor.run({})
    
    # 验证基础提取能力
    print( any(k in ["processing", "language", "human"] for k in result["keywords"]))

def test_textrank_chinese():
    """测试中文TextRank提取及分词"""
    extractor = KeywordExtract(
        text="自然语言处理是人工智能的重要分支，专注于人机交互技术。",
        use_textrank=True,
        max_keywords=2,
        language="chinese"
    )
    result = extractor.run({})
    
    # 验证中文分词效果
    expected_keywords = ["自然语言处理", "人工智能", "人机交互"]
    print( any(k in expected_keywords for k in result["keywords"]))

test_textrank_chinese()
test_textrank_english()