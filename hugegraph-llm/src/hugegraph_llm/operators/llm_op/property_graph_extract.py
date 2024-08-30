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


import json
import re
from typing import List, Any, Dict

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.document.chunk_split import ChunkSplitter
from hugegraph_llm.utils.log import log

# TODO: put in a separate file for users to customize the content
SCHEMA_EXAMPLE_PROMPT = """## Main Task
Given the following graph schema and a piece of text, your task is to analyze the text and extract information that fits into the schema's structure, formatting the information into vertices and edges as specified.

## Basic Rules
### Schema Format
Graph Schema:
- Vertices: [List of vertex labels and their properties]
- Edges: [List of edge labels, their source and target vertex labels, and properties]

### Content Rule
Please read the provided text carefully and identify any information that corresponds to the vertices and edges defined in the schema. For each piece of information that matches a vertex or edge, format it according to the following JSON structures:
#### Vertex Format:
{"id":"vertexLabelID:entityName","label":"vertexLabel","type":"vertex","properties":{"propertyName":"propertyValue",
...}}

#### Edge Format:
{"label":"edgeLabel","type":"edge","outV":"sourceVertexId","outVLabel":"sourceVertexLabel","inV":"targetVertexId","inVLabel":"targetVertexLabel","properties":{"propertyName":"propertyValue",...}}

Also follow the rules: 
1. Don't extract property fields that do not exist in the given schema
2. Ensure the extract property is in the same type as the schema (like 'age' should be a number)
3. If there are multiple primarykeys provided, then the generating strategy of VID is: vertexlabelID:pk1!pk2!pk3 (pk means primary key, and '!' is the separator, no extra space between them)
4. Your output should be a list of such JSON objects, each representing either a vertex or an edge, extracted and formatted based on the text and the provided schema.
5. Translate the given schema filed into Chinese if the given text is Chinese but the schema is in English (Optional) 


## Example
### Input example:
#### text
Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010. James, in his professional life, works as a journalist.  
#### graph schema
{"vertices":[{"vertex_label":"person","properties":["name","age","occupation"]}], "edges":[{"edge_label":"roommate", "source_vertex_label":"person","target_vertex_label":"person","properties":["date"]]}

### Output example:
[{"id":"1:Sarah","label":"person","type":"vertex","properties":{"name":"Sarah","age":30,"occupation":"attorney"}},{"id":"1:James","label":"person","type":"vertex","properties":{"name":"James","occupation":"journalist"}},{"label":"roommate","type":"edge","outV":"1:Sarah","outVLabel":"person","inV":"1:James","inVLabel":"person","properties":{"date":"2010"}}]
"""


def generate_extract_property_graph_prompt(text, schema=None) -> str:
    return f"""---

Following the full instructions above, try to extract the following text from the given schema, output the JSON result:
# Input
## Text:
{text}
## Graph schema
{schema}

# Output"""


def split_text(text: str) -> List[str]:
    chunk_splitter = ChunkSplitter(split_type="paragraph", language="zh")
    chunks = chunk_splitter.split(text)
    return chunks


class PropertyGraphExtract:
    def __init__(
            self,
            llm: BaseLLM,
            example_prompt: str = SCHEMA_EXAMPLE_PROMPT
    ) -> None:
        self.llm = llm
        self.example_prompt = example_prompt
        self.NECESSARY_ITEM_KEYS = {"label", "type", "properties"}  # pylint: disable=invalid-name

    def run(self, context: Dict[str, Any]) -> Dict[str, List[Any]]:
        schema = context["schema"]
        chunks = context["chunks"]
        if "vertices" not in context:
            context["vertices"] = []
        if "edges" not in context:
            context["edges"] = []
        items = []
        for chunk in chunks:
            proceeded_chunk = self.extract_property_graph_by_llm(schema, chunk)
            log.debug("[LLM] %s input: %s \n output:%s", self.__class__.__name__, chunk, proceeded_chunk)
            items.extend(self._extract_and_filter_label(schema, proceeded_chunk))
        items = self.filter_item(schema, items)
        for item in items:
            if item["type"] == "vertex":
                context["vertices"].append(item)
            elif item["type"] == "edge":
                context["edges"].append(item)
        return context

    def extract_property_graph_by_llm(self, schema, chunk):
        prompt = generate_extract_property_graph_prompt(chunk, schema)
        if self.example_prompt is not None:
            prompt = self.example_prompt + prompt
        return self.llm.generate(prompt=prompt)

    def _extract_and_filter_label(self, schema, text):
        # analyze llm generated text to JSON
        json_strings = re.findall(r'(\[.*?])', text, re.DOTALL)
        longest_json = max(json_strings, key=lambda x: len(''.join(x)), default=('', ''))

        longest_json_str = ''.join(longest_json).strip()

        items = []
        try:
            property_graph = json.loads(longest_json_str)
            vertex_label_set = {vertex["name"] for vertex in schema["vertexlabels"]}
            edge_label_set = {edge["name"] for edge in schema["edgelabels"]}
            for item in property_graph:
                if not isinstance(item, dict):
                    log.warning("Invalid property graph item type %s.", type(item))
                    continue
                if not self.NECESSARY_ITEM_KEYS.issubset(item.keys()):
                    log.warning("Invalid item keys %s.", item.keys())
                    continue
                if item["type"] == "vertex" or item["type"] == "edge":
                    if (item["label"] not in vertex_label_set
                            and item["label"] not in edge_label_set):
                        log.warning("Invalid item label %s has been ignored.", item["label"])
                    else:
                        items.append(item)
                else:
                    log.warning("Invalid item type %s has been ignored.", item["type"])
        except json.JSONDecodeError:
            log.critical("Invalid property graph! Please check the extracted JSON data carefully")

        return items

    def filter_item(self, schema, items):
        # filter vertex and edge with invalid properties
        filtered_items = []
        properties_map = {"vertex": {}, "edge": {}}
        for vertex in schema["vertexlabels"]:
            properties_map["vertex"][vertex["name"]] = {
                "primary_keys": vertex["primary_keys"],
                "nullable_keys": vertex["nullable_keys"],
                "properties": vertex["properties"]
            }
        for edge in schema["edgelabels"]:
            properties_map["edge"][edge["name"]] = {
                "properties": edge["properties"]
            }
        log.info("properties_map: %s", properties_map)
        for item in items:
            item_type = item["type"]
            if item_type == "vertex":
                label = item["label"]
                non_nullable_keys = (
                    set(properties_map[item_type][label]["properties"])
                    .difference(set(properties_map[item_type][label]["nullable_keys"])))
                for key in non_nullable_keys:
                    if key not in item["properties"]:
                        item["properties"][key] = "NULL"
            for key, value in item["properties"].items():
                if not isinstance(value, str):
                    item["properties"][key] = str(value)
            filtered_items.append(item)

        return filtered_items
