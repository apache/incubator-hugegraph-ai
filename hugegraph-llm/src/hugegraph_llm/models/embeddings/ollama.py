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


from typing import List

import ollama
from .base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    def __init__(
            self,
            model: str,
            host: str = "127.0.0.1",
            port: int = 11434,
            **kwargs
    ):
        self.model = model
        self.client = ollama.Client(host=f"http://{host}:{port}", **kwargs)
        self.embedding_dimension = None

    def get_text_embedding(
            self,
            text: str
    ) -> List[float]:
        """Comment"""
        return list(self.client.embeddings(model=self.model, prompt=text)["embedding"])
