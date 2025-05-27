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
from typing import Optional, List

from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.common_op.check_schema import CheckSchema
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.index_op.build_gremlin_example_index import BuildGremlinExampleIndex
from hugegraph_llm.operators.index_op.gremlin_example_index_query import GremlinExampleIndexQuery
from hugegraph_llm.operators.llm_op.gremlin_generate import GremlinGenerateSynthesize
from hugegraph_llm.utils.decorators import log_time, log_operator_time, record_rpm


class GremlinGenerator:
    def __init__(self, llm: BaseLLM, embedding: BaseEmbedding):
        self.embedding = []
        self.llm = llm
        self.embedding = embedding
        self.result = None
        self.operators = []

    def clear(self):
        self.operators = []
        return self

    def example_index_build(self, examples):
        self.operators.append(BuildGremlinExampleIndex(self.embedding, examples))
        return self

    def import_schema(self, from_hugegraph=None, from_extraction=None, from_user_defined=None):
        if from_hugegraph:
            self.operators.append(SchemaManager(from_hugegraph))
        elif from_user_defined:
            self.operators.append(CheckSchema(from_user_defined))
        elif from_extraction:
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("No input data / invalid schema type")
        return self

    def example_index_query(self, num_examples):
        self.operators.append(GremlinExampleIndexQuery(self.embedding, num_examples))
        return self

    def gremlin_generate_synthesize(
        self,
        schema,
        gremlin_prompt: Optional[str] = None,
        vertices: Optional[List[str]] = None,
        properties: Optional[List[tuple]] = None
    ):
        self.operators.append(GremlinGenerateSynthesize(self.llm, schema, vertices, gremlin_prompt, properties))
        return self

    def print_result(self):
        self.operators.append(PrintResult())
        return self

    @log_time("total time")
    @record_rpm
    def run(self, **kwargs):
        context = kwargs
        for operator in self.operators:
            context = self._run_operator(operator, context)
        return context

    @log_operator_time
    def _run_operator(self, operator, context):
        return operator.run(context)
