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


from typing import Dict, Any

from hugegraph_llm.llms.base import BaseLLM
from hugegraph_llm.operators.common_op.check_schema import CheckSchema
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import CommitToKg
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.llm_op.disambiguate_data import DisambiguateData
from hugegraph_llm.operators.llm_op.info_extract import InfoExtract


class KgBuilder:
    def __init__(self, llm: BaseLLM):
        self.operators = []
        self.llm = llm
        self.result = None

    def import_schema(self, from_hugegraph=None, from_extraction=None, from_user_defined=None):
        if from_hugegraph:
            self.operators.append(SchemaManager(from_hugegraph))
        elif from_user_defined:
            self.operators.append(CheckSchema(from_user_defined))
        elif from_extraction:
            raise Exception("Not implemented yet")
        else:
            raise Exception("No input data")
        return self

    def extract_triples(self, text: str):
        self.operators.append(InfoExtract(self.llm, text))
        return self

    def disambiguate_word_sense(self):
        self.operators.append(DisambiguateData(self.llm))
        return self

    def commit_to_hugegraph(self):
        self.operators.append(CommitToKg())
        return self

    def print_result(self):
        self.operators.append(PrintResult())
        return self

    def run(self) -> Dict[str, Any]:
        context = None
        for operator in self.operators:
            context = operator.run(context)
        return context
