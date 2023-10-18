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
import os
from src.operators.build_kg_operator import BuildKgOperator
from src.operators.llm.openai_llm import OpenAIChat

if __name__ == "__main__":
    #  If you need a proxy to access OpenAI's API, please set your HTTP proxy here
    os.environ["http_proxy"] = "http://113.54.178.43:7890"
    os.environ["https_proxy"] = "http://113.54.178.43:7890"
    api_key = "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"

    default_llm = OpenAIChat(
        api_key=api_key, model_name="gpt-3.5-turbo-16k", max_tokens=4000
    )
    text = (
        "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010. James, "
        "in his professional life, works as a journalist. Additionally, Sarah is the proud owner of the website "
        "www.sarahsplace.com, while James manages his own webpage, though the specific URL is not mentioned here. "
        "These two individuals, Sarah and James, have not only forged a strong personal bond as roommates but have "
        "also carved out their distinctive digital presence through their respective webpages, showcasing their "
        "varied interests and experiences."
    )
    ops = BuildKgOperator(name="1")
    # build kg with only text
    ops.parse_text_to_data(default_llm).disambiguate_data(
        default_llm
    ).commit_data_to_kg().run(text)
    # build kg with text and schemas
    # nodes_schemas = [
    #     {
    #         "label": "Person",
    #         "primary_key": "name",
    #         "properties": {"age": "int", "name": "text", "occupation": "text"},
    #     },
    #     {
    #         "label": "Webpage",
    #         "primary_key": "name",
    #         "properties": {"name": "text", "url": "text"},
    #     },
    # ]
    # relationships_schemas = [
    #     {
    #         "start": "Person",
    #         "end": "Person",
    #         "type": "roommate",
    #         "properties": {"start": "int"},
    #     },
    #     {"start": "Person", "end": "Webpage", "type": "owns", "properties": {}},
    # ]
    # (
    #     ops.parse_text_to_data_with_schemas(
    #         default_llm, nodes_schemas, relationships_schemas
    #     )
    #     .disambiguate_data(default_llm)
    #     .commit_data_to_kg()
    #     .run(text)
    # )
