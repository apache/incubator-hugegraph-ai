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
