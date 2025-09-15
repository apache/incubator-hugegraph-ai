from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.state.ai_state import WkFlowInput

import json
from PyCGraph import GPipeline

from hugegraph_llm.operators.document_op.chunk_split import ChunkSplitNode
from hugegraph_llm.operators.index_op.build_vector_index import BuildVectorIndexNode
from hugegraph_llm.state.ai_state import WkFlowState


class BuildVectorIndexFlow(BaseFlow):
    def __init__(self):
        pass

    def prepare(self, prepared_input: WkFlowInput, texts):
        prepared_input.texts = texts
        prepared_input.language = "zh"
        prepared_input.split_type = "paragraph"
        return

    def build_flow(self, texts):
        pipeline = GPipeline()
        # prepare for workflow input
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, texts)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        chunk_split_node = ChunkSplitNode()
        build_vector_node = BuildVectorIndexNode()
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(build_vector_node, {chunk_split_node}, "build_vector")

        return pipeline

    def post_deal(self, pipeline=None):
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        return json.dumps(res, ensure_ascii=False, indent=2)
