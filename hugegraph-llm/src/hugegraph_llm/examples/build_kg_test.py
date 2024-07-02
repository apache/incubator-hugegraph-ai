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


from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.kg_construction_task import KgBuilder


if __name__ == "__main__":
    builder = KgBuilder(LLMs().get_llm())

    TEXT = (
        "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with"
        " since 2010. James, in his professional life, works as a journalist. Additionally, Sarah"
        " is the proud owner of the website www.sarahsplace.com, while James manages his own"
        " webpage, though the specific URL is not mentioned here. These two individuals, Sarah and"
        " James, have not only forged a strong personal bond as roommates but have also carved out"
        " their distinctive digital presence through their respective webpages, showcasing their"
        " varied interests and experiences."
    )

    schema = {
        "vertices": [
            {"vertex_label": "person", "properties": ["name", "age", "occupation"]},
            {"vertex_label": "webpage", "properties": ["name", "url"]},
        ],
        "edges": [
            {
                "edge_label": "roommate",
                "source_vertex_label": "person",
                "target_vertex_label": "person",
                "properties": {},
            }
        ],
    }

    (
        builder
        # .import_schema(from_hugegraph="xxx")
        # .import_schema(from_extraction="xxx")
        .import_schema(from_user_defined=schema)
        .print_result()
        .extract_triples(TEXT)
        .print_result()
        # .disambiguate_word_sense()
        # .commit_to_hugegraph()
        .run()
    )
