#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from PyCGraph import GParam, CStatus

from typing import Union, List, Optional, Any


class WkFlowInput(GParam):
    texts: Union[str, List[str]] = None  # texts input used by ChunkSplit Node
    language: str = None  # language configuration used by ChunkSplit Node
    split_type: str = None  # split type used by ChunkSplit Node
    example_prompt: str = None  # need by graph information extract
    schema: str = None  # Schema information requeired by SchemaNode
    graph_name: str = None

    def reset(self, _: CStatus) -> None:
        self.texts = None
        self.language = None
        self.split_type = None
        self.example_prompt = None
        self.schema = None
        self.graph_name = None


class WkFlowState(GParam):
    schema: Optional[str] = None  # schema message
    simple_schema: Optional[str] = None
    chunks: Optional[List[str]] = None
    edges: Optional[List[Any]] = None
    vertices: Optional[List[Any]] = None
    triples: Optional[List[Any]] = None
    call_count: Optional[int] = None

    keywords: Optional[List[str]] = None
    vector_result = None
    graph_result = None
    keywords_embeddings = None

    def setup(self):
        self.schema = None
        self.simple_schema = None
        self.chunks = None
        self.edges = None
        self.vertices = None
        self.triples = None
        self.call_count = None

        self.keywords = None
        self.vector_result = None
        self.graph_result = None
        self.keywords_embeddings = None

        return CStatus()

    def to_json(self):
        """
        Automatically returns a JSON-formatted dictionary of all non-None instance members,
        eliminating the need to manually maintain the member list.

        Returns:
            dict: A dictionary containing non-None instance members and their serialized values.
        """
        # Only export instance attributes (excluding methods and class attributes) whose values are not None
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }
