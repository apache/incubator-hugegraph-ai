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


from typing import Literal, Dict, Any, Optional, Union, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
LANGUAGE_ZH = "zh"
LANGUAGE_EN = "en"
SPLIT_TYPE_DOCUMENT = "document"
SPLIT_TYPE_PARAGRAPH = "paragraph"
SPLIT_TYPE_SENTENCE = "sentence"

class ChunkSplit:
    def __init__(
        self,
        texts: Union[str, List[str]],
        split_type: Literal["document", "paragraph", "sentence"] = SPLIT_TYPE_DOCUMENT,
        language: Literal["zh", "en"] = LANGUAGE_ZH,
    ):
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        self.separators = self._get_separators(language)
        self.text_splitter = self._get_text_splitter(split_type)

    def _get_separators(self, language: str) -> List[str]:
        if language == LANGUAGE_ZH:
            return ["\n\n", "\n", "。", "，", ""]
        if language == LANGUAGE_EN:
            return ["\n\n", "\n", ".", ",", " ", ""]
        raise ValueError("language must be zh or en")

    def _get_text_splitter(self, split_type: str):
        if split_type == SPLIT_TYPE_DOCUMENT:
            return lambda text: [text]
        if split_type == SPLIT_TYPE_PARAGRAPH:
            return RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=30, separators=self.separators
            ).split_text
        if split_type == SPLIT_TYPE_SENTENCE:
            return RecursiveCharacterTextSplitter(
                chunk_size=50, chunk_overlap=0, separators=self.separators
            ).split_text
        raise ValueError("Type must be paragraph, sentence, html or markdown")

    def run(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        all_chunks = []
        for text in self.texts:
            chunks = self.text_splitter(text)
            all_chunks.extend(chunks)

        if context is None:
            return {"chunks": all_chunks}
        context["chunks"] = all_chunks
        return context
