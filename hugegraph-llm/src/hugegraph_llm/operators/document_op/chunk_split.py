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


class ChunkSplit:
    def __init__(
            self,
            texts: Union[str, List[str]],
            split_type: Literal["paragraph", "sentence"] = "paragraph",
            language: Literal["zh", "en"] = "zh"
    ):
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        if language == "zh":
            separators = ["\n\n", "\n", "。", "，", ""]
        elif language == "en":
            separators = ["\n\n", "\n", ".", ",", " ", ""]
        else:
            raise ValueError("language must be zh or en")
        if split_type == "paragraph":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=30,
                separators=separators
            )
        elif split_type == "sentence":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=50,
                chunk_overlap=0,
                separators=separators
            )
        else:
            raise ValueError("type must be paragraph, sentence, html or markdown")

    def run(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        all_chunks = []
        for text in self.texts:
            chunks = self.text_splitter.split_text(text)
            all_chunks.extend(chunks)
        if context is None:
            return {"chunks": all_chunks}
        context["chunks"] = all_chunks
        return context
