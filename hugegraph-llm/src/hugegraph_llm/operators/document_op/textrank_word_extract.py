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
from collections import defaultdict
from typing import Dict

import igraph as ig
import jieba.posseg as pseg
import nltk
import regex

from hugegraph_llm.operators.common_op.nltk_helper import NLTKHelper


class MultiLingualTextRank:
    def __init__(self, keyword_num: int = 5, window_size: int = 3):
        self.top_k = keyword_num
        self.window = window_size if 0 < window_size <= 10 else 3
        self.graph = None
        self.max_len = 100

        self.pos_filter = {
            'chinese': ('n', 'nr', 'ns', 'nt', 'nrt', 'nz', 'v', 'vd', 'vn', "eng", "j", "l"),
            'english': ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBN', 'VBZ')
        }
        self.rules = [r"https?://\S+|www\.\S+",
                      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                      r"\b\w+(?:[-’\']\w+)+\b",
                      r"\b\d+[,.]\d+\b"]

    def _word_mask(self, text):

        placeholder_id_counter = 0
        placeholder_map = {}

        def _create_placeholder(match_obj):
            nonlocal placeholder_id_counter
            original_word = match_obj.group(0)
            _placeholder = f" __shieldword_{placeholder_id_counter}__ "
            placeholder_map[_placeholder.strip()] = original_word
            placeholder_id_counter += 1
            return _placeholder

        special_regex = regex.compile('|'.join(self.rules), regex.V1)
        text = special_regex.sub(_create_placeholder, text)

        return text, placeholder_map

    @staticmethod
    def _get_valid_tokens(masked_text):
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
        return pos_tags

    def _multi_preprocess(self, text):
        # 1. 初始化
        words = []
        ch_tokens = []
        en_stop_words = NLTKHelper().stopwords(lang='english')
        ch_stop_words = NLTKHelper().stopwords(lang='chinese')

        # 2. 屏蔽特殊词
        masked_text, placeholder_map = self._word_mask(text)

        # 3. 清洗过滤标点符号与无效 token
        pos_tags = self._get_valid_tokens(masked_text)

        # 4. 英文分词
        for word, flag in pos_tags:
            # 先检查占位符，如果是占位符则直接替换为原单词输出
            if word in placeholder_map:
                words.append(placeholder_map[word])
            else:
                if len(word) >= 1 and flag in self.pos_filter['english'] and word.lower() not in en_stop_words:
                    # 存在中文字符会重新分词，否则加入分词
                    words.append(word)
                    if re.compile('[\u4e00-\u9fff]').search(word):
                        ch_tokens.append(word)

        # 5. 需要进一步的话，中文分词
        if len(ch_tokens) > 0:
            ch_tokens = list(set(ch_tokens))
            for ch_token in ch_tokens:
                idx = words.index(ch_token)
                ch_words = []
                jieba_tokens = pseg.cut(ch_token)
                for word, flag in jieba_tokens:
                    if len(word) >= 1 and flag in self.pos_filter['chinese'] and word not in ch_stop_words:
                        ch_words.append(word)
                words = words[:idx] + ch_words + words[idx+1:]

        return words

    def _build_graph(self, words):
        unique_words = list(set(words))
        name_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        edge_weights = defaultdict(int)
        for i, word1 in enumerate(words):
            for j in range(i + 1, min(i + self.window + 1, len(words))):
                word2 = words[j]
                if word1 != word2:
                    pair = tuple(sorted((word1, word2)))
                    edge_weights[pair] += 1

        graph = ig.Graph(n=len(unique_words), directed=False)
        graph.vs['name'] = unique_words
        edges_idx = [(name_to_idx[a], name_to_idx[b]) for (a, b) in edge_weights.keys()]
        graph.add_edges(edges_idx)
        graph.es['weight'] = list(edge_weights.values())
        self.graph = graph

    def _rank_nodes(self):
        if not self.graph or self.graph.vcount() == 0:
            return {}

        pagerank_scores = self.graph.pagerank(directed=False, damping=0.85, weights='weight')
        if max(pagerank_scores) > 0:
            pagerank_scores = [scores/max(pagerank_scores) for scores in pagerank_scores]
        node_names = self.graph.vs['name']
        return dict(zip(node_names, pagerank_scores))

    def extract_keywords(self, text) -> Dict[str, float]:
        # 1. nltk 模型载入
        if not NLTKHelper().check_nltk_data():
            return {}

        # 2. 文本预处理
        words = self._multi_preprocess(text)
        if not words:
            return {}

        # 3. 构建图，运行 PageRank 算法
        unique_words = list(dict.fromkeys(words))
        ranks = dict(zip(unique_words, [0] * len(unique_words)))
        if len(unique_words) > 1:
            self._build_graph(words)
            if not self.graph or self.graph.vcount() == 0:
                return {}
            ranks = self._rank_nodes()
        return ranks
