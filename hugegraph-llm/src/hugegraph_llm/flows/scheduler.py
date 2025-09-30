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

import threading
from typing import Dict, Any
from PyCGraph import GPipeline, GPipelineManager
from hugegraph_llm.flows.build_vector_index import BuildVectorIndexFlow
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.flows.graph_extract import GraphExtractFlow
from hugegraph_llm.flows.import_graph_data import ImportGraphDataFlow
from hugegraph_llm.flows.update_vid_embeddings import UpdateVidEmbeddingsFlows
from hugegraph_llm.flows.get_graph_index_info import GetGraphIndexInfoFlow
from hugegraph_llm.flows.build_schema import BuildSchemaFlow
from hugegraph_llm.flows.prompt_generate import PromptGenerateFlow
from hugegraph_llm.flows.rag_flow_raw import RAGRawFlow
from hugegraph_llm.flows.rag_flow_vector_only import RAGVectorOnlyFlow
from hugegraph_llm.flows.rag_flow_graph_only import RAGGraphOnlyFlow
from hugegraph_llm.flows.rag_flow_graph_vector import RAGGraphVectorFlow
from hugegraph_llm.state.ai_state import WkFlowInput
from hugegraph_llm.utils.log import log
from hugegraph_llm.flows.text2gremlin import Text2GremlinFlow


class Scheduler:
    pipeline_pool: Dict[str, Any] = None
    max_pipeline: int

    def __init__(self, max_pipeline: int = 10):
        self.pipeline_pool = {}
        # pipeline_pool act as a manager of GPipelineManager which used for pipeline management
        self.pipeline_pool["build_vector_index"] = {
            "manager": GPipelineManager(),
            "flow": BuildVectorIndexFlow(),
        }
        self.pipeline_pool["graph_extract"] = {
            "manager": GPipelineManager(),
            "flow": GraphExtractFlow(),
        }
        self.pipeline_pool["import_graph_data"] = {
            "manager": GPipelineManager(),
            "flow": ImportGraphDataFlow(),
        }
        self.pipeline_pool["update_vid_embeddings"] = {
            "manager": GPipelineManager(),
            "flow": UpdateVidEmbeddingsFlows(),
        }
        self.pipeline_pool["get_graph_index_info"] = {
            "manager": GPipelineManager(),
            "flow": GetGraphIndexInfoFlow(),
        }
        self.pipeline_pool["build_schema"] = {
            "manager": GPipelineManager(),
            "flow": BuildSchemaFlow(),
        }
        self.pipeline_pool["prompt_generate"] = {
            "manager": GPipelineManager(),
            "flow": PromptGenerateFlow(),
        }
        self.pipeline_pool["text2gremlin"] = {
            "manager": GPipelineManager(),
            "flow": Text2GremlinFlow(),
        }
        # New split rag pipelines
        self.pipeline_pool["rag_raw"] = {
            "manager": GPipelineManager(),
            "flow": RAGRawFlow(),
        }
        self.pipeline_pool["rag_vector_only"] = {
            "manager": GPipelineManager(),
            "flow": RAGVectorOnlyFlow(),
        }
        self.pipeline_pool["rag_graph_only"] = {
            "manager": GPipelineManager(),
            "flow": RAGGraphOnlyFlow(),
        }
        self.pipeline_pool["rag_graph_vector"] = {
            "manager": GPipelineManager(),
            "flow": RAGGraphVectorFlow(),
        }
        self.max_pipeline = max_pipeline

    # TODO: Implement Agentic Workflow
    def agentic_flow(self):
        pass

    def schedule_flow(self, flow: str, *args, **kwargs):
        if flow not in self.pipeline_pool:
            raise ValueError(f"Unsupported workflow {flow}")
        manager: GPipelineManager = self.pipeline_pool[flow]["manager"]
        flow: BaseFlow = self.pipeline_pool[flow]["flow"]
        pipeline: GPipeline = manager.fetch()
        if pipeline is None:
            # call coresponding flow_func to create new workflow
            pipeline = flow.build_flow(*args, **kwargs)
            status = pipeline.init()
            if status.isErr():
                error_msg = f"Error in flow init: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            status = pipeline.run()
            if status.isErr():
                error_msg = f"Error in flow execution: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            res = flow.post_deal(pipeline)
            manager.add(pipeline)
            return res
        else:
            # fetch pipeline & prepare input for flow
            prepared_input = pipeline.getGParamWithNoEmpty("wkflow_input")
            flow.prepare(prepared_input, *args, **kwargs)
            status = pipeline.run()
            if status.isErr():
                error_msg = f"Error in flow execution {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            res = flow.post_deal(pipeline)
            manager.release(pipeline)
            return res

    async def schedule_stream_flow(self, flow: str, *args, **kwargs):
        if flow not in self.pipeline_pool:
            raise ValueError(f"Unsupported workflow {flow}")
        manager: GPipelineManager = self.pipeline_pool[flow]["manager"]
        flow: BaseFlow = self.pipeline_pool[flow]["flow"]
        pipeline: GPipeline = manager.fetch()
        if pipeline is None:
            # call coresponding flow_func to create new workflow
            pipeline = flow.build_flow(*args, **kwargs)
            try:
                pipeline.getGParamWithNoEmpty("wkflow_input").stream = True
                status = pipeline.init()
                if status.isErr():
                    error_msg = f"Error in flow init: {status.getInfo()}"
                    log.error(error_msg)
                    raise RuntimeError(error_msg)
                status = pipeline.run()
                if status.isErr():
                    error_msg = f"Error in flow execution: {status.getInfo()}"
                    log.error(error_msg)
                    raise RuntimeError(error_msg)
                async for res in flow.post_deal_stream(pipeline):
                    yield res
            finally:
                manager.add(pipeline)
        else:
            try:
                # fetch pipeline & prepare input for flow
                prepared_input: WkFlowInput = pipeline.getGParamWithNoEmpty(
                    "wkflow_input"
                )
                prepared_input.stream = True
                flow.prepare(prepared_input, *args, **kwargs)
                status = pipeline.run()
                if status.isErr():
                    raise RuntimeError(f"Error in flow execution {status.getInfo()}")
                async for res in flow.post_deal_stream(pipeline):
                    yield res
            finally:
                manager.release(pipeline)


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
