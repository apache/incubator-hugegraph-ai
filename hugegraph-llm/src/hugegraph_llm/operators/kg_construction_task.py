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


from hugegraph_llm.llms.base import BaseLLM
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import (
    CommitDataToKg,
    CommitSPOToKg,
)
from hugegraph_llm.operators.llm_op.disambiguate_data import DisambiguateData
from hugegraph_llm.operators.llm_op.info_extract import InfoExtract


class KgConstructionTask:
    def __init__(self, llm: BaseLLM):
        self.operators = []
        self.llm = llm
        self.result = None

    def info_extract(self, text: str, nodes_schemas=None, relationships_schemas=None):
        if nodes_schemas and relationships_schemas:
            self.operators.append(InfoExtract(self.llm, text, nodes_schemas, relationships_schemas))
        else:
            self.operators.append(InfoExtract(self.llm, text))
        return self

    def spo_triple_extract(self, text: str):
        self.operators.append(InfoExtract(self.llm, text, spo=True))
        return self

    def word_sense_disambiguation(self, with_schemas=False):
        self.operators.append(DisambiguateData(self.llm, with_schemas))
        return self

    def commit_to_hugegraph(self, spo=False):
        if spo:
            self.operators.append(CommitSPOToKg())
        else:
            self.operators.append(CommitDataToKg())
        return self

    def print_result(self):
        self.operators.append(PrintResult())
        return self

    def run(self):
        result = ""
        for operator in self.operators:
            result = operator.run(result)
