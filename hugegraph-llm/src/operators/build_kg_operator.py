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
from src.operators.build_kg.commit_data_to_kg import CommitDataToKg
from src.operators.build_kg.disambiguate_data import DisambiguateData
from src.operators.build_kg.parse_text_to_data import (
    ParseTextToData,
    ParseTextToDataWithSchemas,
)


class BuildKgOperator:
    def __init__(self, name):
        self.name = name
        self.parse_text_to_kg = []

    def parse_text_to_data(self, llm):
        self.parse_text_to_kg.append(ParseTextToData(llm=llm))
        return self

    def parse_text_to_data_with_schemas(
        self, llm, nodes_schemas, relationships_schemas
    ):
        self.parse_text_to_kg.append(
            ParseTextToDataWithSchemas(
                llm=llm,
                nodes_schema=nodes_schemas,
                relationships_schemas=relationships_schemas,
            )
        )
        return self

    def disambiguate_data(self, llm):
        self.parse_text_to_kg.append(DisambiguateData(llm=llm))
        return self

    def commit_data_to_kg(self):
        self.parse_text_to_kg.append(CommitDataToKg())
        return self

    def run(self, result):
        for i in self.parse_text_to_kg:
            result = i.run(result)
