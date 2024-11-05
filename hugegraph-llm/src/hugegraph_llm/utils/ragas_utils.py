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

from pysbd import Segmenter
from ragas.metrics import (
    ContextEntityRecall,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NoiseSensitivity,
    ResponseRelevancy,
)

RAGAS_METRICS_DICT = {
    "context_entity_recall": ContextEntityRecall(),
    "factual_correctness": FactualCorrectness(),
    "faithfulness": Faithfulness(),
    "llm_context_precision_without_reference": LLMContextPrecisionWithoutReference(),
    "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
    "llm_context_recall": LLMContextRecall(),
    "noise_sensitivity": NoiseSensitivity(),
    "response_relevancy": ResponseRelevancy(),
}

RAGAS_METRICS_ZH_DICT = {
    "context_entity_recall": ContextEntityRecall(),
    "factual_correctness": FactualCorrectness(sentence_segmenter=Segmenter(language="zh", clean=True)),
    "faithfulness": Faithfulness(sentence_segmenter=Segmenter(language="zh", clean=True)),
    "llm_context_precision_without_reference": LLMContextPrecisionWithoutReference(),
    "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
    "llm_context_recall": LLMContextRecall(),
    "noise_sensitivity": NoiseSensitivity(sentence_segmenter=Segmenter(language="zh", clean=True)),
    "response_relevancy": ResponseRelevancy(),
}

