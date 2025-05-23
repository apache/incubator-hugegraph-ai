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

from textrank4zh import TextRank4Keyword
import spacy
import pytextrank
# from transformers import BertForMaskedLM, BertTokenizer

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
    ):
        self._llm = llm
        self._query = text
        self._language = language.lower()
        self._max_keywords = max_keywords
        self._extract_template = extract_template or KEYWORDS_EXTRACT_TPL
        self._use_textrank = use_textrank  # 新增TextRank开关


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
        if self._language == "chinese":
            # 中文使用textrank4zh
            # allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
            allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
            tr4w = TextRank4Keyword(allow_speech_tags=allow_speech_tags)
            try:
                tr4w.analyze(text=self._query, lower=True, window=2, vertex_source='all_filters', edge_source='no_stop_words', pagerank_config={'alpha': 0.85,})
                keywords = [keyword.word for keyword in tr4w.get_keywords(self._max_keywords, word_min_len=2)]
            except Exception as e:
                log.error(f"TextRank提取失败: {str(e)}")
                keywords = []
        if self._language == "english":
            # 英文使用pytextrank
            try:
                nlp = spacy.load("en_core_web_sm")
                nlp.add_pipe("textrank")
                doc = nlp(self._query)
                keywords = [token.text for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]
                keywords = list(set(keywords))[:self._max_keywords]
            except Exception as e:
                log.error(f"TextRank提取失败: {str(e)}")
                keywords = []
        
        log.debug(f"TextRank提取耗时: {time.perf_counter()-start_time:.2f}s")
    
        return set(filter(None, keywords))

    #ToDO: 同义词替换
    # def _expand_keywords_with_synonyms(self, keywords: Set[str]) -> Set[str]:
    #     """ 使用BERT进行关键词同义词替换 """
    #     if not keywords:
    #         return set()
    #     # 加载预训练的BERT模型和分词器
    #     model_name = "bert-base-uncased"
    #     tokenizer = BertTokenizer.from_pretrained(model_name)
    #     model = BertForMaskedLM.from_pretrained(model_name)
    #     model.eval()
    #     expanded_keywords = set()
    #     for keyword in keywords:
    #         # 分词并标记
    #         tokens = tokenizer.tokenize(keyword)
    #         if len(tokens) < 2:
    #             expanded_keywords.add(keyword)
    #             continue
    #         # 替换每个词
    #         for i in range(len(tokens)):
    #             masked_tokens = tokens.copy()
    #             masked_tokens[i] = "[MASK]"
    #             masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    #             input_ids = tokenizer.encode(masked_text, return_tensors="pt")
    #             with torch.no_grad():
    #                 outputs = model(input_ids)  # 模型输出
    #             predictions = outputs.logits[0, i]  # 预测概率
    #             predicted_token_id = torch.argmax(predictions).item()
    #             predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
    #             # 替换并还原
    #             expanded_keywords.add(tokenizer.convert_tokens_to_string(
    #                 [token if token != "[MASK]" else predicted_token for token in masked_tokens]
    #             ))
    #     return expanded_keywords
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
        text="No one’s born being good at all things. You become good at things through hard work. You’re not a varsity athlete " \
        "the first time you play a new sport. You don’t hit every note the first time you sing a song.You’ve got to practice. The "\
        "same principle applies to your schoolwork. You might have to do a math problem a few times before you get it right. You "\
        "might have to read something a few times before you understand it.You definitely have to do a few drafts of a paper before it’s good enough to hand in.",
        use_textrank=True,
        max_keywords=3,
        language="english"
    )
    result = extractor.run({})
    
    # 验证基础提取能力
    print( any(k in ["hard work", "practice", "schoolwork", "repetition", "perseverance"] for k in result["keywords"]))

def test_textrank_chinese():
    """测试中文TextRank提取及分词"""
    extractor = KeywordExtract(
        text="来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。",
        use_textrank=True,
        max_keywords=5,
        language="chinese"
    )
    result = extractor.run({})
    
    # 验证中文分词效果
    expected_keywords = ["小行星", "命名", "国家", "周又元", "天文台"]
    print( any(k in expected_keywords for k in result["keywords"]))

test_textrank_chinese()
test_textrank_english()