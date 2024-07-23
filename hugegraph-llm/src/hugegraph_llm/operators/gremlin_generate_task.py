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


import time

from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.index_op.build_gremlin_example_index import BuildGremlinExampleIndex
from hugegraph_llm.operators.index_op.gremlin_example_index_query import GremlinExampleIndexQuery
from hugegraph_llm.operators.llm_op.gremlin_generate import GremlinGenerate
from hugegraph_llm.utils.log import log


class GremlinGenerator:
    def __init__(self, llm: BaseLLM, embedding: BaseEmbedding):
        self.embedding = []
        self.llm = llm
        self.embedding = embedding
        self.result = None
        self.operators = []

    def example_index_build(self, examples):
        self.operators.append(BuildGremlinExampleIndex(self.embedding, examples))
        return self

    def example_index_query(self, query, num_examples):
        self.operators.append(GremlinExampleIndexQuery(query, self.embedding, num_examples))
        return self

    def gremlin_generate(self, use_schema, use_example, schema):
        self.operators.append(GremlinGenerate(self.llm, use_schema, use_example, schema))
        return self

    def print_result(self):
        self.operators.append(PrintResult())
        return self

    def run(self):
        context = {}
        for operator in self.operators:
            log.debug("Running operator: %s", operator.__class__.__name__)
            start = time.time()
            context = operator.run(context)
            log.debug("Operator %s finished in %s seconds", operator.__class__.__name__,
                      time.time() - start)
            log.debug("Context:\n%s", context)
        return context
