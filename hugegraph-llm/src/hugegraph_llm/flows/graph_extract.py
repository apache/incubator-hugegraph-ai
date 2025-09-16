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

import json
from PyCGraph import GPipeline
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.operators.common_op.check_schema import CheckSchemaNode
from hugegraph_llm.operators.document_op.chunk_split import ChunkSplitNode
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManagerNode
from hugegraph_llm.operators.llm_op.info_extract import InfoExtractNode
from hugegraph_llm.operators.llm_op.property_graph_extract import (
    PropertyGraphExtractNode,
)
from hugegraph_llm.utils.log import log


class GraphExtractFlow(BaseFlow):
    def __init__(self):
        pass

    def _import_schema(
        self,
        from_hugegraph=None,
        from_extraction=None,
        from_user_defined=None,
    ):
        if from_hugegraph:
            return SchemaManagerNode()
        elif from_user_defined:
            return CheckSchemaNode()
        elif from_extraction:
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("No input data / invalid schema type")

    def prepare(
        self, prepared_input: WkFlowInput, schema, texts, example_prompt, extract_type
    ):
        # prepare input data
        prepared_input.texts = texts
        prepared_input.language = "zh"
        prepared_input.split_type = "document"
        prepared_input.example_prompt = example_prompt
        prepared_input.schema = schema
        schema = schema.strip()
        if schema.startswith("{"):
            try:
                schema = json.loads(schema)
                prepared_input.schema = schema
            except json.JSONDecodeError as exc:
                log.error("Invalid JSON format in schema. Please check it again.")
                raise ValueError("Invalid JSON format in schema.") from exc
        else:
            log.info("Get schema '%s' from graphdb.", schema)
            prepared_input.graph_name = schema
        return

    def build_flow(self, schema, texts, example_prompt, extract_type):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        # prepare input data
        self.prepare(prepared_input, schema, texts, example_prompt, extract_type)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")
        schema = schema.strip()
        schema_node = None
        if schema.startswith("{"):
            try:
                schema = json.loads(schema)
                schema_node = self._import_schema(from_user_defined=schema)
            except json.JSONDecodeError as exc:
                log.error("Invalid JSON format in schema. Please check it again.")
                raise ValueError("Invalid JSON format in schema.") from exc
        else:
            log.info("Get schema '%s' from graphdb.", schema)
            schema_node = self._import_schema(from_hugegraph=schema)

        chunk_split_node = ChunkSplitNode()
        graph_extract_node = None
        if extract_type == "triples":
            graph_extract_node = InfoExtractNode()
        elif extract_type == "property_graph":
            graph_extract_node = PropertyGraphExtractNode()
        else:
            raise ValueError(f"Unsupported extract_type: {extract_type}")
        pipeline.registerGElement(schema_node, set(), "schema_node")
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(
            graph_extract_node, {schema_node, chunk_split_node}, "graph_extract"
        )

        return pipeline

    def post_deal(self, pipeline=None):
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        vertices = res.get("vertices", [])
        edges = res.get("edges", [])
        if not vertices and not edges:
            log.info("Please check the schema.(The schema may not match the Doc)")
            return json.dumps(
                {
                    "vertices": vertices,
                    "edges": edges,
                    "warning": "The schema may not match the Doc",
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"vertices": vertices, "edges": edges},
            ensure_ascii=False,
            indent=2,
        )
