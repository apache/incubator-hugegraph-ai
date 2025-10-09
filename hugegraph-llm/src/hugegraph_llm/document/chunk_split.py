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


from typing import Literal, Union, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkSplitter:
    def __init__(
<<<<<<< HEAD
        self,
        split_type: Literal["paragraph", "sentence"] = "paragraph",
        language: Literal["zh", "en"] = "zh",
=======
        self, split_type: Literal["paragraph", "sentence"] = "paragraph", language: Literal["zh", "en"] = "zh"
>>>>>>> 87ee5d3 (style: format code with black line-length 120)
    ):
        if language == "zh":
            separators = ["\n\n", "\n", "。", "，", ""]
        elif language == "en":
            separators = ["\n\n", "\n", ".", ",", " ", ""]
        else:
            raise ValueError("Argument `language` must be zh or en!")
        if split_type == "paragraph":
<<<<<<< HEAD
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=30, separators=separators
            )
        elif split_type == "sentence":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=50, chunk_overlap=0, separators=separators
            )
=======
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30, separators=separators)
        elif split_type == "sentence":
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0, separators=separators)
>>>>>>> 87ee5d3 (style: format code with black line-length 120)
        else:
            raise ValueError("Arg `type` must be paragraph, sentence!")

    def split(self, documents: Union[str, List[str]]) -> List[str]:
        chunks = []
        if isinstance(documents, str):
            documents = [documents]
        for document in documents:
            chunks.extend(self.text_splitter.split_text(document))
        return chunks
