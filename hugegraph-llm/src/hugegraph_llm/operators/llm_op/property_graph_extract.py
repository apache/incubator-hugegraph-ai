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


SCHEMA_EXAMPLE_PROMPT = """Main Task
Given the following graph schema and a piece of text, your task is to analyze the text and extract information that fits into the schema’s structure, formatting the information into vertices and edges as specified.

Basic Rules
Schema Format
Graph Schema:

Vertices: [List of vertex labels and their properties]
Edges: [List of edge labels, their source and target vertex labels, and properties]
Content Rule
Please read the provided text carefully and identify any information that corresponds to the vertices and edges defined in the schema. For each piece of information that matches a vertex or edge, format it according to the following JSON structures:

Vertex Format:
{“label”:“vertexLabel”,“type”:“vertex”,“properties”:{“propertyName”:“propertyValue”,…}}

Edge Format:
{“label”:“edgeLabel”,“type”:“edge”,“outV”:“sourceVertexId”,“outVLabel”:“sourceVertexLabel”,“inV”:“targetVertexId”,“inVLabel”:“targetVertexLabel”,“properties”:{“propertyName”:“propertyValue”,…}}

Also follow the rules:

Don’t extract attribute/property fields that do not exist in the given schema
Ensure the extract property is in the same type as the schema (like ‘age’ should be a number)
Translate the given schema filed into Chinese if the given text is Chinese but the schema is in English (Optional)
Your output should be a list of such JSON objects, each representing either a vertex or an edge, extracted and formatted based on the text and the provided schema.
PrimaryKey ID Generate Rule

vertexLabel的id生成策略为：id:primaryKey1!primaryKey2

Example
Input example:
text
道路交通事故认定书
鱼公交认字[2013]第00478号
天气：小雨
交通事故时间：2013车11月24日18时09分
交通事故地点：251省避清河菜市场路口
当事人、车辆、道路和交通环境等基本情况：
1、当事人基本情况：
张小虎，男，1972年1月3日出生，山东省鱼台县清河镇清河村62号，系驾驶鲁H72886号小型轿车，驾驶证号：370827197201032316，档案号：370800767691，准驾车型：C1E,电话：15606376419.
于海洋，男，1952年3月12日出生，山东省鱼台县号清河镇于屯村77号、身份证：370827195203122316,步行，电话：15092699426。
2、车辆情况：
鲁H7Z886小型轿车，入户车主：谢彪。有交通事故责任强制保险。保险单号：PDZA20133708T000075766,保险公司：中国人民产保险股份有限公司济宁市分公司。
3、道路和咬通环境等基本情况：
事故现场位于251省道鱼台县清河镇菜市场路口，251省道呈南北走向，道路平坦，沥青路面，视线一般，有交通标志、标线，有中心隔离带，两侧为商业店铺
道路交通事故发生经过：
2013年日月24日18时09分，张小虎驾驶鲁H72886号小型斩车，沿251省道自北向南行业至鱼台县清河镇菜市场路口处时与自西向东步行过公路的于海洋相撞，致于海洋受伤入院，经鱼台县人民医院抢教无效，于洋于2013车11月27日死亡，车辆损坏造成道路交通事故。张小虎肇事后驾车逃逸。
道略交通事故证据及事故形成原因分折：
根据现场勘查、当事人陈述证实：张小虎因观察不够，措施不当违反《中华人民共和国道路交通安全法》第三十八条“车辆、行人应当按照交通信号通行：遇有交通警察现场指挥时，应当按照交道警察的指挥通行：在没有交通信号的道路上，应当在确保安全、畅通的原则下通行。”之规定，因酒后驾车，违反《中华人民共和国道路交通安全法》第二十二条第二款“饮酒，服用国家管制的精神药品或者醉药品，或者患有妨碍安全驾驶杭动车的疾病，或者过度劳影响安全驾驶的，不得买驶机动车，”是事故发生的原因，且肇事后驾车逸逸。
当事人导致交通事故的过错及责任或者意外原因：
根据《中华人民共和国道路交通全法实施条例》第九十二条第一款和《道路交通事故处理程序规定》第四十六条的规定，认定当事人张小虎担地次事教的全部贡任。当事人于海洋无责任。
交通警察：
刘爱军HZ402
二0一四年一月二日


graph schema
{"vertexLabels":[{"id":3,"name":"法条","id_strategy":"PRIMARY_KEY","primary_keys":["法典名","法条索引"],"nullable_keys":["法章名","法条内容"],"properties":["法典名","法章名","法条索引","法条内容"]},{"id":7,"name":"事故","id_strategy":"PRIMARY_KEY","primary_keys":["事故认定书编号","事故认定书单位"],"nullable_keys":[],"properties":["事故发生时间","事故认定书编号","事故认定书单位"]},{"id":11,"name":"发生地点","id_strategy":"PRIMARY_KEY","primary_keys":["城市","所属路段"],"nullable_keys":["走向","材质","路面情况","道路状况"],"properties":["城市","走向","材质","路面情况","道路状况","所属路段"]},{"id":12,"name":"当事人","id_strategy":"PRIMARY_KEY","primary_keys":["身份证号"],"nullable_keys":["姓名","性别","年龄","民族","驾照"],"properties":["身份证号","姓名","性别","年龄","民族","驾照"]},{"id":13,"name":"车辆","id_strategy":"PRIMARY_KEY","primary_keys":["车辆牌照"],"nullable_keys":["行驶证所属人","保险公司","保险情况","车辆类型"],"properties":["车辆牌照","行驶证所属人","保险公司","保险情况","车辆类型"]},{"id":14,"name":"行为","id_strategy":"PRIMARY_KEY","primary_keys":["行为名称"],"nullable_keys":[],"properties":["行为名称"]}],"edgeLabels":[{"id":7,"name":"事故相关法条","source_label":"事故","target_label":"法条","sort_keys":[],"nullable_keys":[],"properties":[]},{"id":8,"name":"事故相关当事人","source_label":"事故","target_label":"当事人","sort_keys":[],"nullable_keys":["责任认定"],"properties":["责任认定"]},{"id":9,"name":"事故相关行为","source_label":"事故","target_label":"行为","sort_keys":[],"nullable_keys":[],"properties":[]},{"id":10,"name":"当事人相关行为","source_label":"当事人","target_label":"行为","sort_keys":[],"nullable_keys":[],"properties":[]},{"id":11,"name":"当事人相关车辆","source_label":"当事人","target_label":"车辆","sort_keys":[],"nullable_keys":[],"properties":[]},{"id":12,"name":"事故发生地点","source_label":"事故","target_label":"发生地点","sort_keys":[],"nullable_keys":[],"properties":[]}]}

Output example:
[{"label":"事故","type":"vertex","properties":{"事故发生时间":"2013-11-24 18:09:00.000","事故认定书编号":"鱼公交认字[2013]第00478号","事故认定书单位":"道路交通事故认定书"}},{"label":"发生地点","type":"vertex","properties":{"城市":"山东省鱼台县","所属路段":"251省道清河菜市场路口","走向":"南北","材质":"沥青","路面情况":"平坦","道路状况":"视线一般"}},{"label":"当事人","type":"vertex","properties":{"身份证号":"370827197201032316","姓名":"张小虎","性别":"男","年龄":"1972-01-03","驾照":"C1E"}},{"label":"当事人","type":"vertex","properties":{"身份证号":"370827195203122316","姓名":"于海洋","性别":"男","年龄":"1952-03-12"}},{"label":"车辆","type":"vertex","properties":{"车辆牌照":"鲁H7Z886","行驶证所属人":"谢彪","保险公司":"中国人民产保险股份有限公司济宁市分公司","保险情况":"交通事故责任强制保险","车辆类型":"小型轿车"}},{"label":"行为","type":"vertex","properties":{"行为名称":"逃逸"}},{"label":"行为","type":"vertex","properties":{"行为名称":"酒后驾车"}},{"label":"行为","type":"vertex","properties":{"行为名称":"观察不够"}},{"label":"法条","type":"vertex","properties":{"法典名":"中华人民共和国道路交通安全法","法条索引":"第三十八条","法条内容":"车辆、行人应当按照交通信号通行；遇有交通警察现场指挥时，应当按照交通警察的指挥通行；在没有交通信号的道路上，应当在确保安全、畅通的原则下通行。"}},{"label":"法条","type":"vertex","properties":{"法典名":"中华人民共和国道路交通安全法","法条索引":"第二十二条","法条内容":"饮酒，服用国家管制的精神药品或者醉药品，或者患有妨碍安全驾驶杭动车的疾病，或者过度劳影响安全驾驶的，不得买驶机动车。"}},{"label":"事故相关法条","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"事故","inV":"3:中华人民共和国道路交通安全法!第三十八条","inVLabel":"法条","properties":{}},{"label":"事故相关法条","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"事故","inV":"3:中华人民共和国道路交通安全法!第二十二条","inVLabel":"法条","properties":{}},{"label":"事故相关当事人","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"事故","inV":"12: 370827197201032316","inVLabel":"当事人","properties":{"责任认定":"全部责任"}},{"label":"事故相关当事人","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"事故","inV":"12: 370827195203122316","inVLabel":"当事人","properties":{"责任认定":"无责任"}},{"label":"事故相关行为","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"当事人","inV":"14:逃逸","inVLabel":"行为","properties":{}},{"label":"事故相关行为","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"当事人","inV":"14:酒后驾车","inVLabel":"行为","properties":{}},{"label":"事故相关行为","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"当事人","inV":"14:观察不够","inVLabel":"行为","properties":{}},{"label":"当事人相关行为","type":"edge","outV":"12:370827197201032316","outVLabel":"当事人","inV":"14:逃逸","inVLabel":"行为","properties":{}},{"label":"当事人相关行为","type":"edge","outV":"12:370827197201032316","outVLabel":"当事人","inV":"14:酒后驾车","inVLabel":"行为","properties":{}},{"label":"当事人相关行为","type":"edge","outV":"12:370827197201032316","outVLabel":"当事人","inV":"14:观察不够","inVLabel":"行为","properties":{}},{"label":"当事人相关车辆","type":"edge","outV":"12:370827197201032316","outVLabel":"当事人","inV":"13:鲁H7Z886","inVLabel":"车辆","properties":{}},{"label":"事故发生地点","type":"edge","outV":"7:鱼公交认字[2013]第00478号!道路交通事故认定书","outVLabel":"事故","inV":"11:山东省鱼台县!251省道清河菜市场路口","inVLabel":"发生地点","properties":{}}]
"""


def generate_extract_property_graph_prompt(text, schema=None) -> str:
    return f"""---

请根据上面的完整指令, 尝试根据下面给定的 schema, 提取下面的文本, 只需要输出 json 结果:
## Text:
{text}
## Graph schema:
{schema}"""


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
        self.NECESSARY_ITEM_KEYS = {"label", "type", "properties"}

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
            log.debug("[LLM] input: %s \n output:%s", chunk, proceeded_chunk)
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
        # analyze llm generated text to json
        json_strings = re.findall(r'(\[.*?])', text, re.DOTALL)
        longest_json = max(json_strings, key=lambda x: len(''.join(x)), default=('', ''))

        longest_json_str = ''.join(longest_json).strip()

        items = []
        try:
            property_graph = json.loads(longest_json_str)
            vertex_label_set = set([vertex["vertex_label"] for vertex in schema["vertices"]])
            edge_label_set = set([edge["edge_label"] for edge in schema["edges"]])
            for item in property_graph:
                if not isinstance(item, dict):
                    log.warning("Invalid property graph item type %s.", type(item))
                    continue
                if not self.NECESSARY_ITEM_KEYS.issubset(item.keys()):
                    log.warning("Invalid item keys %s.", item.keys())
                    continue
                if item["type"] == "vertex" or item["type"] == "edge":
                    if item["label"] not in vertex_label_set and item["label"] not in edge_label_set:
                        log.warning("Invalid item label %s has been ignored.", item["label"])
                    else:
                        items.append(item)
                else:
                    log.warning("Invalid item type %s has been ignored.", item["type"])
        except json.JSONDecodeError:
            log.error("Invalid property graph!")

        return items

    def filter_item(self, schema, items):
        # filter vertex and edge with invalid properties
        filtered_items = []
        properties_map = {"vertex": {}, "edge": {}}
        for vertex in schema["vertices"]:
            properties_map["vertex"][vertex["vertex_label"]] = {
                "primary_keys": vertex["primary_keys"],
                "nullable_keys": vertex["nullable_keys"],
                "properties": vertex["properties"]
            }
        for edge in schema["edges"]:
            properties_map["edge"][edge["edge_label"]] = {
                "properties": edge["properties"]
            }
        log.info("properties_map: %s", properties_map)
        for item in items:
            item_type = item["type"]
            if item_type == "vertex":
                label = item["label"]
                non_nullable_keys = set(properties_map[item_type][label]["properties"]).difference(set(properties_map[item_type][label]["nullable_keys"]))
                for key in non_nullable_keys:
                    if key not in item["properties"]:
                        item["properties"][key] = "NULL"
            for key, value in item["properties"].items():
                if not isinstance(value, str):
                    item["properties"][key] = str(value)
            filtered_items.append(item)

        # filter vertex with the same id or primary keys
        # items = filtered_items
        # filtered_items = []
        # vertex_ids = set()
        # vertex_primary_key_set = set()
        # for item in items:
        #     if item["type"] == "vertex":
        #         label = item["label"]
        #         if item["id"] in vertex_ids:
        #             log.warning("Duplicate vertex id %s has been ignored.", item["id"])
        #             continue
        #         vertex_ids.add(item["id"])
        #         primary_key_tuple = (label,)
        #         for key in properties_map["vertex"][label]["primary_keys"]:
        #             value = item["properties"][key]
        #             primary_key_tuple += (key, value)
        #         if primary_key_tuple in vertex_primary_key_set:
        #             log.warning("Duplicate vertex primary key %s has been ignored.", primary_key_tuple)
        #             continue
        #         vertex_primary_key_set.add(primary_key_tuple)
        #         filtered_items.append(item)
        #     else:
        #         filtered_items.append(item)

        # filter edge with invalid source or target
        # items = filtered_items
        # filtered_items = []
        # vertex_ids = set([item["id"] for item in items if item["type"] == "vertex"])
        # for item in items:
        #     if item["type"] == "edge":
        #         if item["source"] not in vertex_ids or item["target"] not in vertex_ids:
        #             log.warning("Invalid edge source or target %s has been ignored.", item)
        #             continue
        #         filtered_items.append(item)
        return filtered_items
