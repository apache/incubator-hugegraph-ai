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

from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    answer_relevancy,
    context_recall,
    context_utilization,
    context_entity_recall,
)

RAGAS_METRICS_DICT = {
    "context_precision": context_precision,
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "answer_correctness": answer_correctness,
    "context_recall": context_recall,
    "context_utilization": context_utilization,
    "context_entity_recall": context_entity_recall,
}
