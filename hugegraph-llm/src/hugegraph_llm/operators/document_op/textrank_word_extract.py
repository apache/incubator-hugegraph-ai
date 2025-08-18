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
import os
import re
from collections import defaultdict

import igraph as ig
import jieba.posseg as pseg
import nltk
import regex

from hugegraph_llm.utils.anchor import get_project_root
from hugegraph_llm.utils.log import log

EXTRACT_STOPWORDS = 'hugegraph_llm.resources.nltk_data.corpora.stopwords'
nltk.data.path.insert(0, os.path.join(get_project_root(), 'src/hugegraph_llm/resources/nltk_data'))


class MultiLingualTextRank:
    def __init__(self, keyword_num: int = 5, window_size: int = 2, mask_words: str = ""):
        self.top_k = keyword_num
        self.window = window_size
        self.graph = None
        self.max_len = 100

        self.pos_filter = {
            'chinese': ('n', 'nr', 'ns', 'nt', 'nrt', 'nz', 'v', 'vd', 'vn', "eng"),
            'english': ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBN', 'VBZ')
        }

        self.stopwords = {'chinese': set(), 'english': set()}
        self.stopwords_loaded = False
        self.mask_words = list(filter(None, (mask_words or "").split(',')))
        self.chinese_pattern = re.compile('[\u4e00-\u9fa5]')

    def _load_stopwords(self):
        if self.stopwords_loaded:
            return True
        resource_path = importlib.resources.files(EXTRACT_STOPWORDS)
        try:
            with resource_path.joinpath('chinese').open(encoding='utf-8') as f:
                self.stopwords['chinese'] = {line.strip() for line in f}
        except FileNotFoundError:
            log.error("Chinese stopwords file not found, using empty set")
            return False
        try:
            with resource_path.joinpath('english').open(encoding='utf-8') as f:
                self.stopwords['english'] = {line.strip() for line in f}
        except FileNotFoundError:
            log.error("English stopwords file not found, using empty set")
            return False
        return True

    @staticmethod
    def _build_word_regex(word: str):
        return rf'(?<![a-zA-Z0-9])({re.escape(word)})(?![a-zA-Z0-9])'

    @staticmethod
    def _regex_test(pattern: str, text: str):
        pattern = pattern[1:-1].strip()
        if len(pattern) == 0:
            return None
        try:
            test_pattern = regex.compile(pattern, regex.V1)
            test_pattern.search(text, timeout=1)
            return pattern
        except (regex.error, OverflowError, TimeoutError) as e:
            log.error("Failed to compile or test pattern '%s...': %s", pattern, e)
            return None

    def _load_maskwords(self, text):
        mask_patterns = [
            r'https?://\S+|www\.\S+',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\w+(?:[-’\']\w+)+\b',
            r'\b\d+[,.]\d+\b',
        ]
        for word in self.mask_words:
            word = word.strip()
            if not isinstance(word, str) or len(word) > self.max_len or not word:
                continue
            if word.startswith('/') and word.endswith('/'):
                pattern = self._regex_test(word, text)
                if pattern:
                    mask_patterns.append(pattern)
            else:
                mask_patterns.append(self._build_word_regex(word))

        return mask_patterns

    def _multi_preprocess(self, text):
        # 1. 初始化
        words = []
        ch_tokens = []
        en_stop_words = self.stopwords.get('english', set())
        ch_stop_words = self.stopwords.get('chinese', set())

        # 2. 屏蔽特殊词
        placeholder_id_counter = 0
        placeholder_map = {}

        def _create_placeholder(match_obj):
            nonlocal placeholder_id_counter
            original_word = match_obj.group(0)
            _placeholder = f" __shieldword_{placeholder_id_counter}__ "
            placeholder_map[_placeholder.strip()] = original_word
            placeholder_id_counter += 1
            return _placeholder

        mask_patterns = self._load_maskwords(text)
        special_regex = regex.compile('|'.join(mask_patterns))
        masked_text = special_regex.sub(_create_placeholder, text)

        # 3. 保留词设置，清洗过滤标点符号，进行分层
        patterns_to_keep = [
            r'__shieldword_\d+__',
            r'\b\w+\b',
            r'[\u4e00-\u9fff]+'
        ]
        combined_pattern = re.compile('|'.join(patterns_to_keep), re.IGNORECASE)
        tokens = combined_pattern.findall(masked_text)
        text_for_nltk = ' '.join(tokens)
        nltk_tokens = nltk.word_tokenize(text_for_nltk)
        pos_tags = nltk.pos_tag(nltk_tokens)

        # 4. 英文分词
        for word, flag in pos_tags:
            # 先检查占位符，如果是占位符则直接替换为原单词输出
            if word in placeholder_map:
                words.append(placeholder_map[word])
            else:
                if len(word) > 1 and flag in self.pos_filter['english'] and word not in en_stop_words:
                    # 存在中文字符会重新分词，否则加入分词
                    words.append(word)
                    if self.chinese_pattern.search(word):
                        ch_tokens.append(word)

        # 5. 中文分词
        ch_tokens = list(set(ch_tokens))
        for ch_token in ch_tokens:
            idx = words.index(ch_token)
            ch_words = []
            jieba_tokens = pseg.cut(ch_token)
            for word, flag in jieba_tokens:
                if len(word) > 1 and flag in self.pos_filter['chinese'] and word not in ch_stop_words:
                    ch_words.append(word)
            words = words[:idx] + ch_words + words[idx+1:]

        return list(set(words))

    def _build_graph(self, words):
        unique_words = list(set(words))
        edge_weights = defaultdict(int)
        for i, word1 in enumerate(words):
            for j in range(i + 1, i + self.window):
                if j < len(words):
                    word2 = words[j]
                    if word1 != word2:
                        pair = tuple(sorted((word1, word2)))
                        edge_weights[pair] += 1

        graph = ig.Graph(directed=False)
        graph.add_vertices(unique_words)

        edges = list(edge_weights.keys())
        weights = list(edge_weights.values())

        graph.add_edges(edges)
        graph.es['weight'] = weights

        self.graph = graph

    def _rank_nodes(self):
        if not self.graph or self.graph.vcount() == 0:
            return {}

        pagerank_scores = self.graph.pagerank(directed=False, damping=0.85, weights='weight')
        node_names = self.graph.vs['name']
        return dict(zip(node_names, pagerank_scores))

    def extract_keywords(self, text):
        # 1. 停止词载入
        if not self._load_stopwords():
            return []

        # 2. 文本预处理
        words = self._multi_preprocess(text)
        if not words:
            return []

        # 3. 构建图，运行 PageRank 算法
        unique_words = list(set(words))
        if len(unique_words) > self.window:
            self._build_graph(words)
            if not self.graph or self.graph.vcount() == 0:
                return []
            ranks = self._rank_nodes()
            top_keywords = sorted(ranks, key=ranks.get, reverse=True)[:self.top_k]
        else:
            top_keywords = unique_words

        return top_keywords
