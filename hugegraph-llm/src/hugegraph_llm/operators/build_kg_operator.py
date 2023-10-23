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
from hugegraph_llm.operators.hugegraph_op.commit_data_to_kg import CommitDataToKg
from hugegraph_llm.operators.llm_op.disambiguate_data import DisambiguateData
from hugegraph_llm.operators.llm_op.parse_text_to_data import (
    ParseTextToData,
    ParseTextToDataWithSchemas,
)


class KgBuilder:
    def __init__(self, llm: BaseLLM):
        self.parse_text_to_kg = []
        self.llm = llm
        self.data = {}

    def parse_text_to_data(self, text: str):
        self.parse_text_to_kg.append(ParseTextToData(llm=self.llm, text=text))
        return self

    def parse_text_to_data_with_schemas(self, text: str, nodes_schemas, relationships_schemas):
        self.parse_text_to_kg.append(
            ParseTextToDataWithSchemas(
                llm=self.llm,
                text=text,
                nodes_schema=nodes_schemas,
                relationships_schemas=relationships_schemas,
            )
        )
        return self

    def disambiguate_data(self):
        self.parse_text_to_kg.append(DisambiguateData(llm=self.llm, is_user_schema=False))
        return self

    def disambiguate_data_with_schemas(self):
        self.parse_text_to_kg.append(DisambiguateData(llm=self.llm, is_user_schema=True))
        return self

    def commit_data_to_kg(self):
        self.parse_text_to_kg.append(CommitDataToKg())
        return self

    def run(self):
        result = ""
        for i in self.parse_text_to_kg:
            result = i.run(result)
            print(result)
