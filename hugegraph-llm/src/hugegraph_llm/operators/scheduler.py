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


from PyCGraph import GPipeline

from hugegraph_llm.operators.common_op.check_schema import CheckSchemaNode
from hugegraph_llm.operators.document_op.chunk_split import ChunkSplitNode
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManagerNode
from hugegraph_llm.operators.index_op.build_vector_index import BuildVectorIndexNode
from hugegraph_llm.state.ai_state import WkFlowState, WkFlowInput

import json

from ..utils.log import log

from hugegraph_llm.operators.llm_op.info_extract import InfoExtractNode
from hugegraph_llm.operators.llm_op.property_graph_extract import (
    PropertyGraphExtractNode,
)


class Scheduler:
    def __init__(self):
        pass

    # TODO: Implement Agentic Workflow
    def agentic_flow():
        pass

    def _import_schema(
        self,
        prepared_input: WkFlowInput,
        from_hugegraph=None,
        from_extraction=None,
        from_user_defined=None,
    ):
        if from_hugegraph:
            prepared_input.graph_name = from_hugegraph
            return SchemaManagerNode()
        elif from_user_defined:
            prepared_input.schema = from_user_defined
            return CheckSchemaNode()
        elif from_extraction:
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("No input data / invalid schema type")

    # Fixed flow
    def build_vector_index_flow(self, texts):
        pipeline = GPipeline()
        # prepare for workflow input
        prepared_input = WkFlowInput()
        prepared_input.texts = texts
        prepared_input.language = "zh"
        prepared_input.split_type = "paragraph"

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        chunk_split_node = ChunkSplitNode()
        build_vector_node = BuildVectorIndexNode()
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(build_vector_node, {chunk_split_node}, "build_vector")

        pipeline.init()
        status = pipeline.run()
        pipeline.destroy()
        print(f"pipeline status {status.getInfo()}")
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        return json.dumps(res, ensure_ascii=False, indent=2)

    # 固定流程：图谱抽取
    def graph_extract_flow(self, schema, texts, example_prompt, extract_type):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        # prepare input data
        prepared_input.texts = texts
        prepared_input.language = "zh"
        prepared_input.split_type = "document"
        prepared_input.example_prompt = example_prompt
        prepared_input.schema = schema

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")
        schema = schema.strip()
        schema_node = None
        if schema.startswith("{"):
            try:
                schema = json.loads(schema)
                schema_node = self._import_schema(
                    prepared_input=prepared_input, from_user_defined=schema
                )
            except json.JSONDecodeError:
                log.error("Invalid JSON format in schema. Please check it again.")
                return (
                    "ERROR: Invalid JSON format in schema. Please check it carefully."
                )
        else:
            log.info("Get schema '%s' from graphdb.", schema)
            schema_node = self._import_schema(
                prepared_input=prepared_input, from_hugegraph=schema
            )

        chunk_split_node = ChunkSplitNode()
        graph_extract_node = None
        if extract_type == "triples":
            graph_extract_node = InfoExtractNode()
        elif extract_type == "property_graph":
            graph_extract_node = PropertyGraphExtractNode()
        pipeline.registerGElement(schema_node, set(), "chunk_split")
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(
            graph_extract_node, {schema_node, chunk_split_node}, "graph_extract"
        )

        status = pipeline.process()
        print(f"pipeline status {status.getInfo()}")
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        if not res["vertices"] and not res["edges"]:
            log.info("Please check the schema.(The schema may not match the Doc)")
            return json.dumps(
                {
                    "vertices": res["vertices"],
                    "edges": res["edges"],
                    "warning": "The schema may not match the Doc",
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"vertices": res["vertices"], "edges": res["edges"]},
            ensure_ascii=False,
            indent=2,
        )


class SchedulerSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Scheduler()
        return cls._instance
