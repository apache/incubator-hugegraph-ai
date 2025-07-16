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

import importlib.resources
import re
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Set

import jieba
import jieba.posseg as pseg
import networkx as nx
import nltk

from hugegraph_llm.config import prompt
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper
from hugegraph_llm.utils.log import log

KEYWORDS_EXTRACT_TPL = prompt.keywords_extract_prompt

EXTRACT_STOPWORDS = 'hugegraph_llm.resources.nltk_data.corpora.stopwords'


class KeywordExtract:
    def __init__(
        self,
        text: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        max_keywords: int = 5,
        extract_template: Optional[str] = None,
        language: str = "en",
        extract_method: str = "TextRank",  # 新增关键词提取方法设置
        textrank_kwargs: Optional[Dict] = None,  # TextRank 参数
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL
        self._extract_method = extract_method  # 新增关键词提取方法设置
        self._textrank_model = MultiLingualTextRank(**textrank_kwargs)  # TextRank 参数

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

        if self._extract_method == "TextRank":
            # 使用 TextRank 提取关键词
            keywords = self._extract_with_textrank()
        else:
            # 使用 LLM 提取关键词
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
        """ TextRank 提取模式 """
        start_time = time.perf_counter()
        try:
            keywords = self._textrank_model.extract_keywords(self._query, self._language)
        except FileNotFoundError as e:
            log.error("TextRank resource file not found: %s", e)
            keywords = []
        except (TypeError, ValueError) as e:
            log.error("TextRank parameter error: %s", e)
            keywords = []
        except MemoryError as e:
            log.error("TextRank memory error (text too large?): %s", e)
            keywords = []
        log.debug("TextRank Keyword extraction time: %.2fs",
                  time.perf_counter() - start_time)
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
                results.update(
                    w for w in sub_tokens if w not in NLTKHelper().stopwords(lang=self._language))
        return results


class MultiLingualTextRank:
    def __init__(self, keyword_num=5, window_size=5, mask_words=""):
        self.top_k = keyword_num
        self.window = window_size
        self.graph = None

        # 定义中英文的候选词性
        self.pos_filter = {
            'zh': ('n', 'nr', 'ns', 'nt', 'nrt', 'nz', 'v', 'vd', 'vn', "eng"),
            'en': ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBN', 'VBZ')
        }

        self.stopwords = {'zh': set(), 'en': set()}

        # 定义特殊词列表，支持用户传入自定义特殊词，防止中文分词时切分特殊单词

        self.mask_words = list(filter(None, (mask_words or "").split(',')))

        self.stopwords_loaded = False

    def _load_stopwords(self):
        if self.stopwords_loaded:
            return True
        resource_path = importlib.resources.files(EXTRACT_STOPWORDS)
        try:
            with resource_path.joinpath('chinese').open(encoding='utf-8') as f:
                self.stopwords['zh'] = {line.strip() for line in f}
        except FileNotFoundError:
            log.error("Chinese stopwords file not found, using empty set")
            return False
        try:
            with resource_path.joinpath('english').open(encoding='utf-8') as f:
                self.stopwords['en'] = {line.strip() for line in f}
        except FileNotFoundError:
            log.error("English stopwords file not found, using empty set")
            return False
        return True

    def _preprocess(self, text, lang):
        """
        - 'en': 清理、分词、标注、过滤。
        - 'zh': 遮蔽特殊词 -> 占位符与过滤词输入 -> 清理 -> 中文分词 -> 恢复特殊词 -> 过滤。
        """
        words = []
        if lang.startswith('zh'):

            # 1. 遮蔽 (Masking)
            placeholder_id_counter = 0
            placeholder_map = {}

            def _create_placeholder(match_obj):
                nonlocal placeholder_id_counter
                original_word = match_obj.group(0)
                _placeholder = f"__SPECIAL_TOKEN_{placeholder_id_counter}__"
                placeholder_map[_placeholder] = original_word
                placeholder_id_counter += 1
                return _placeholder

            # 特殊词与短语匹配模式
            all_patterns = [
                r'(?:https?://|www\.)[\w\-\.\/\:?=&%#]+',
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                r'\b\w+(?:[-’\']\w+)+\b',
                r'\b\d+[,.]\d+\b',
            ]
            if self.mask_words:
                escaped_words = [re.escape(word) for word in self.mask_words]
                mask_words_pattern = r'(?<![a-zA-Z0-9])(' + '|'.join(
                    escaped_words) + r')(?![a-zA-Z0-9])'
                all_patterns = [mask_words_pattern] + all_patterns

            special_regex = re.compile('|'.join(all_patterns))
            masked_text = special_regex.sub(_create_placeholder, text)

            # 2. 清理 (Cleaning)
            final_patterns_to_keep = [
                r'__SPECIAL_TOKEN_\d+__',
                r'\b\w+\b',
                r'[\u4e00-\u9fff]+'
            ]
            final_token_regex = re.compile('|'.join(final_patterns_to_keep))
            clean_tokens = final_token_regex.findall(masked_text)
            text_for_jieba = ' '.join(clean_tokens)

            # 3. 在分词前，将所有占位符作为一个完整的词添加到 jieba 词典中
            for placeholder in placeholder_map:
                jieba.add_word(placeholder, tag='SPTK')

            # 4. 分词与恢复
            stop_words = self.stopwords.get('zh', set())
            jieba_tokens = pseg.cut(text_for_jieba)

            for word, flag in jieba_tokens:
                if word in placeholder_map:
                    restored_word = placeholder_map[word]
                    words.append(restored_word)
                else:
                    if len(word) > 1 and flag in self.pos_filter['zh'] and word not in stop_words:
                        words.append(word)

        elif lang.startswith('en'):

            all_patterns_to_keep = [
                r'https?://\S+|www\.\S+',
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                r'\b\w+(?:[-’\']\w+)+\b',
                r'\b\d+[,.]\d+\b',
                r'\b\w+\b'
            ]
            combined_pattern = re.compile('|'.join(all_patterns_to_keep), re.IGNORECASE)
            tokens = combined_pattern.findall(text)
            text_for_nltk = ' '.join(tokens)

            stop_words = self.stopwords.get('en', set())
            text_for_nltk = text_for_nltk.lower()
            nltk_tokens = nltk.word_tokenize(text_for_nltk)
            pos_tags = nltk.pos_tag(nltk_tokens)

            for word, flag in pos_tags:
                if len(word) > 1 and flag in self.pos_filter['en'] and word not in stop_words:
                    words.append(word)
        return words

    def _build_graph(self, words):
        """
        构建词共现图（修改后）
        """
        self.graph = nx.Graph()
        unique_words = set(words)
        self.graph.add_nodes_from(unique_words)

        if len(unique_words) < self.window:
            return

        edge_weights = defaultdict(int)
        for i, word1 in enumerate(words):
            for j in range(i + 1, i + self.window):
                if j < len(words):
                    word2 = words[j]
                    # 确保两个词不相同，避免自环
                    if word1 != word2:
                        edge_weights[(word1, word2)] += 1

        for (word1, word2), weight in edge_weights.items():
            self.graph.add_edge(word1, word2, weight=weight)

    def _rank_nodes(self):
        """
        运行 PageRank 算法
        """
        # 如果图中没有节点，直接返回空字典
        if not self.graph or not self.graph.nodes():
            return {}
        return nx.pagerank(self.graph, alpha=0.85, weight='weight')

    def extract_keywords(self, text, lang):
        """
        主函数：执行完整的关键词提取流程
        """
        # 1. 停止词载入
        if not self._load_stopwords():
            return []

        # 2. 文本预处理
        words = self._preprocess(text, lang)
        if not words:
            return []

        # 3. 构建图
        self._build_graph(words)

        # 4. 运行 TextRank
        if not self.graph or self.graph.number_of_nodes() == 0:
            return []
        ranks = self._rank_nodes()

        # 5. 提取 Top-K 关键词
        top_keywords = sorted(ranks, key=ranks.get, reverse=True)[:self.top_k]

        return top_keywords
