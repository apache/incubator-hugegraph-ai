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


from typing import Dict, Any, List

import jieba
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from nltk.translate.bleu_score import sentence_bleu


def get_score(query: str, content: str) -> float:
    query_tokens = jieba.lcut(query)
    content_tokens = jieba.lcut(content)
    return sentence_bleu([query_tokens], content_tokens)


class MergeDedupRerank:
    def __init__(self, embedding: BaseEmbedding, topk: int = 10):
        self.embedding = embedding
        self.topk = topk

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: 逻辑应该是：
        # 1. 分词后优先关键词匹配 vid （直接匹配）
        # 2. 匹配不到，尝试退化模糊vid召回（并提示用户是否想问的是xxx）
        # 3. 之后我们可以把chunk/graph加上，查询的时候可以同时查询关键词对应的chunk，然后一起综合（进阶）
        query = context.get("query")

        vector_result = context.get("vector_result", [])
        vector_result = self._dedup_and_rerank(query, vector_result)[:self.topk]

        graph_result = context.get("graph_result", [])
        graph_result = self._dedup_and_rerank(query, graph_result)[:self.topk]

        context["vector_result"] = vector_result
        context["graph_result"] = graph_result

        return context

    def _dedup_and_rerank(self, query: str, results: List[str]):
        results = list(set(results))
        result_score_list = [[res, get_score(query, res)] for res in results]
        result_score_list.sort(key=lambda x: x[1], reverse=True)
        return [res[0] for res in result_score_list]
