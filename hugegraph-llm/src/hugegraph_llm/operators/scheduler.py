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
import threading
from typing import Dict, Any
from PyCGraph import GPipeline, GPipelineManager

from hugegraph_llm.operators.common_op.check_schema import CheckSchemaNode
from hugegraph_llm.operators.document_op.chunk_split import ChunkSplitNode
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManagerNode
from hugegraph_llm.operators.index_op.build_vector_index import BuildVectorIndexNode
from hugegraph_llm.state.ai_state import WkFlowState, WkFlowInput
from hugegraph_llm.operators.llm_op.info_extract import InfoExtractNode
from hugegraph_llm.operators.llm_op.property_graph_extract import (
    PropertyGraphExtractNode,
)
from hugegraph_llm.utils.log import log


class Scheduler:
    pipeline_pool: Dict[str, Any] = None
    max_pipeline: int

    def __init__(self, max_pipeline: int = 10):
        self.pipeline_pool = {}
        # pipeline_pool act as a manager of GPipelineManager which used for pipeline management
        self.pipeline_pool["build_vector_index"] = {
            "manager": GPipelineManager(),
            "flow_func": self.build_vector_index_flow,
            "prepare_func": self.build_vector_index_prepare,
            "post_func": self.build_vector_index_post,
        }
        self.pipeline_pool["graph_extract"] = {
            "manager": GPipelineManager(),
            "flow_func": self.graph_extract_flow,
            "prepare_func": self.graph_extract_prepare,
            "post_func": self.graph_extract_post,
        }
        self.max_pipeline = max_pipeline

    # TODO: Implement Agentic Workflow
    def agentic_flow(self):
        pass

    def schedule_flow(self, flow: str, *args, **kwargs):
        if flow not in self.pipeline_pool:
            raise "Unsupported workflow"
        manager = self.pipeline_pool[flow]["manager"]
        flow_func = self.pipeline_pool[flow]["flow_func"]
        prepare_func = self.pipeline_pool[flow]["prepare_func"]
        post_func = self.pipeline_pool[flow]["post_func"]
        pipeline = manager.fetch()
        if pipeline is None:
            # call coresponding flow_func to create new workflow
            pipeline = flow_func(*args, **kwargs)
            status = pipeline.init()
            if status.isErr():
                raise "Error in flow init"
            status = pipeline.run()
            if status.isErr():
                raise "Error in flow execution"
            res = post_func(pipeline)
            manager.add(pipeline)
            return res
        else:
            # fetch pipeline & prepare input for flow
            prepared_input = pipeline.getGParamWithNoEmpty("wkflow_input")
            prepare_func(prepared_input, *args, **kwargs)
            status = pipeline.run()
            if status.isErr():
                raise f"Error in flow execution {status.getInfo()}"
            res = post_func(pipeline)
            manager.release(pipeline)
            return res

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

    def build_vector_index_prepare(self, prepared_input: WkFlowInput, texts):
        prepared_input.texts = texts
        prepared_input.language = "zh"
        prepared_input.split_type = "paragraph"
        return

    # Fixed flow: build vector index
    def build_vector_index_flow(self, texts):
        pipeline = GPipeline()
        # prepare for workflow input
        prepared_input = WkFlowInput()
        self.build_vector_index_prepare(prepared_input, texts)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        chunk_split_node = ChunkSplitNode()
        build_vector_node = BuildVectorIndexNode()
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(build_vector_node, {chunk_split_node}, "build_vector")

        return pipeline

    def build_vector_index_post(self, pipeline):
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        return json.dumps(res, ensure_ascii=False, indent=2)

    def graph_extract_prepare(
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
            except json.JSONDecodeError:
                log.error("Invalid JSON format in schema. Please check it again.")
                raise (
                    "ERROR: Invalid JSON format in schema. Please check it carefully."
                )
        else:
            log.info("Get schema '%s' from graphdb.", schema)
            prepared_input.graph_name = schema
        return

    # Fixed flow: graph extraction
    def graph_extract_flow(self, schema, texts, example_prompt, extract_type):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        # prepare input data
        self.graph_extract_prepare(
            prepared_input, schema, texts, example_prompt, extract_type
        )

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")
        schema = schema.strip()
        schema_node = None
        if schema.startswith("{"):
            try:
                schema = json.loads(schema)
                schema_node = self._import_schema(from_user_defined=schema)
            except json.JSONDecodeError:
                log.error("Invalid JSON format in schema. Please check it again.")
                raise (
                    "ERROR: Invalid JSON format in schema. Please check it carefully."
                )
        else:
            log.info("Get schema '%s' from graphdb.", schema)
            schema_node = self._import_schema(from_hugegraph=schema)

        chunk_split_node = ChunkSplitNode()
        graph_extract_node = None
        if extract_type == "triples":
            graph_extract_node = InfoExtractNode()
        elif extract_type == "property_graph":
            graph_extract_node = PropertyGraphExtractNode()
        pipeline.registerGElement(schema_node, set(), "schema_node")
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(
            graph_extract_node, {schema_node, chunk_split_node}, "graph_extract"
        )

        return pipeline

    def graph_extract_post(self, pipeline):
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
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = Scheduler()
        return cls._instance
