# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded per-chunk extraction, review and repair with ordered concurrent batches."""

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hugegraph_llm.models.llms.base import LLMOutputTruncatedError

log = logging.getLogger(__name__)


class ExtractionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_reviews: int = Field(default=2, ge=0, le=5, strict=True)
    max_fixes: int = Field(default=1, ge=0, le=3, strict=True)
    max_workers: int = Field(default=4, ge=1, le=8, strict=True)


@dataclass
class ChunkInput:
    index: int
    text: str


@dataclass
class Issue:
    path: str
    message: str
    evidence: str | None = None


@dataclass
class ReviewResult:
    decision: Literal["pass", "fix", "block"]
    issues: list[Issue] = field(default_factory=list)


@dataclass
class ChunkResult:
    index: int
    status: Literal["final", "candidate", "blocked", "failed"]
    graph: dict | None
    issues: list[Issue]
    reason: str
    stage: str | None
    reviews_used: int
    fixes_used: int
    llm_calls: int

    def as_dict(self) -> dict:
        return asdict(self)


def run_chunk(*, bundle, chunk: ChunkInput, options: ExtractionOptions) -> ChunkResult:
    """Extract, validate, review and repair one chunk with local business counters."""
    current_graph: dict | None = None
    issues: list[Issue] = []
    reviews_used = fixes_used = llm_calls = 0
    stage = "extract"

    def finish(
        status: Literal["final", "candidate", "blocked", "failed"], reason: str, failure_stage: str | None = None
    ) -> ChunkResult:
        return ChunkResult(
            index=chunk.index,
            status=status,
            graph=current_graph,
            issues=issues,
            reason=reason,
            stage=failure_stage,
            reviews_used=reviews_used,
            fixes_used=fixes_used,
            llm_calls=llm_calls,
        )

    try:
        log.debug("Chunk %s: extract", chunk.index)
        llm_calls += 1
        current_graph = bundle.extract(chunk)
        while True:
            stage = "validate"
            issues = bundle.validate(current_graph, chunk)
            if issues:
                if fixes_used >= options.max_fixes:
                    return finish("blocked", "validation_failed")
            else:
                if reviews_used >= options.max_reviews:
                    return finish("candidate", "review_budget_exhausted")
                stage = "review"
                reviews_used += 1
                llm_calls += 1
                log.debug("Chunk %s: review %s", chunk.index, reviews_used)
                review = bundle.review(current_graph, chunk)
                issues = review.issues
                if review.decision == "pass":
                    return finish("final", "review_passed")
                if review.decision == "block":
                    return finish("blocked", "review_blocked")
                if review.decision != "fix":
                    raise ValueError(f"Unsupported review decision: {review.decision!r}")
                if fixes_used >= options.max_fixes:
                    return finish("candidate", "fix_budget_exhausted")

            stage = "repair"
            fixes_used += 1
            llm_calls += 1
            log.debug("Chunk %s: repair %s", chunk.index, fixes_used)
            current_graph = bundle.repair(current_graph, chunk, issues)
            issues = []
    except LLMOutputTruncatedError as exc:
        log.warning("Chunk %s output truncated during %s (%s)", chunk.index, stage, exc.reason)
        return finish("failed", f"{stage}_output_truncated: {exc.reason}", stage)
    except Exception as exc:  # A chunk failure must not discard other chunks' results.
        log.warning("Chunk %s failed during %s (%s)", chunk.index, stage, type(exc).__name__)
        return finish("failed", f"{stage}_failed: {type(exc).__name__}", stage)


def run_chunks(
    *,
    chunks: Iterable[ChunkInput],
    prepare: Callable[[ChunkInput], object],
    options: ExtractionOptions,
) -> tuple[ChunkResult, ...]:
    """Create a fresh bundle per chunk and collect failures alongside successes.

    Preparation runs inside the worker limit. The finite batch is submitted and
    retained in memory; input iteration failures propagate after workers finish.
    """

    def run_prepared_chunk(chunk: ChunkInput) -> ChunkResult:
        try:
            bundle = prepare(chunk)
        except Exception as exc:
            return ChunkResult(
                index=chunk.index,
                status="failed",
                graph=None,
                issues=[],
                reason=f"prepare_failed: {type(exc).__name__}",
                stage="prepare",
                reviews_used=0,
                fixes_used=0,
                llm_calls=0,
            )
        return run_chunk(bundle=bundle, chunk=chunk, options=options)

    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        return tuple(executor.map(run_prepared_chunk, chunks))
