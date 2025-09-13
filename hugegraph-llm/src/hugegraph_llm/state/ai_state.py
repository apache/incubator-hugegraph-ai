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

from typing import Union, List


class WkFlowInput(GParam):
    texts: Union[str, List[str]] = None  # texts input used by ChunkSplit Node
    language: str = None  # language configuration used by ChunkSplit Node
    split_type: str = None  # split type used by ChunkSplit Node
    # huge_settings: HugeGraphConfig = None
    # llm_settings: LLMConfig = None
    # resource_path: str = None # needed by BuildVectorIndex Node
    example_prompt: str = None  # need by graph information extract
    schema: str = None  # Schema information requeired by SchemaNode
    graph_name: str = None

    def reset(self, curStatus: CStatus):
        self.texts = None
        self.language = None
        self.split_type = None
        # self.huge_settings = None
        # self.llm_settings = None
        # self.resource_path = None
        self.example_prompt = None
        self.schema = None
        self.graph_name = None
        return


class WkFlowState(GParam):
    schema: str = ""  # schema message
    simple_schema: str = ""
    chunks: list[str] = []
    edges: list
    vertices: list
    triples: list
    call_count: int

    keywords: list[str] = []
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
        自动返回所有非None成员的JSON格式化字典，无需手动维护成员列表

        Returns:
            dict: 包含非None成员及其序列化值的字典
        """
        # 只导出实例属性（排除方法和类属性），且值不为None
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }
