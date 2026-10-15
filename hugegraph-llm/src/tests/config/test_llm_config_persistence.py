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


"""Cross-config races with real dotenv persistence and deterministic interleavings."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from unittest.mock import Mock

import pytest
import requests
from dotenv import dotenv_values
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from hugegraph_llm import config
from hugegraph_llm.api import rag_api
from hugegraph_llm.config.llm_config import LLMConfig
from hugegraph_llm.config.models import base_config
from hugegraph_llm.demo.rag_demo import configs_block

pytestmark = pytest.mark.contract
ROLES = ("chat", "extract", "text2gql")


class ObservedLock:
    """Keep real locking semantics and notify only when a caller must wait."""

    def __init__(self, competing_progress):
        self.lock = RLock()
        self.competing_progress = competing_progress

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.competing_progress.set()
            self.lock.acquire()
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    # model_construct avoids BaseConfig initialization and its environment-file
    # side effects. Only this test-owned object and file receive mutations.
    settings = LLMConfig.model_construct()
    for field in settings.model_dump():
        if field.endswith("api_key"):
            setattr(settings, field, "synthetic-old-key")
    for role in ROLES:
        setattr(settings, f"{role}_llm_type", "openai")
        for provider in ("openai", "litellm"):
            setattr(settings, f"{provider}_{role}_api_base", "http://old.invalid/v1")
            setattr(settings, f"{provider}_{role}_language_model", "old-model")
        setattr(settings, f"ollama_{role}_language_model", "old-model")
    settings.embedding_type = "openai"
    settings.openai_embedding_api_base = "http://old-embedding.invalid/v1"
    settings.openai_embedding_model = "old-embedding"
    settings.reranker_type = "cohere"
    settings.reranker_model = "old-reranker"
    settings.cohere_base_url = "http://old-reranker.invalid/v1/rerank"
    for module in (config, rag_api, configs_block):
        monkeypatch.setattr(module, "llm_settings", settings)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(base_config, "env_path", str(env_file))
    monkeypatch.setattr(configs_block.gr, "Info", lambda *_args: None)
    return settings, env_file


def seed_file(settings, env_file):
    env_file.write_text(
        "".join(
            f"{key.upper()}={value if value is not None else ''}\n" for key, value in settings.model_dump().items()
        ),
        encoding="utf-8",
    )


def make_client():
    router = APIRouter()
    rag_api.rag_http_api(
        router,
        Mock(),
        Mock(),
        Mock(),
        configs_block.apply_llm_config,
        configs_block.apply_embedding_config,
        configs_block.apply_reranker_config,
        Mock(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def fail_llm(client, entry):
    if entry == "http":
        return client.post(
            "/config/llm",
            json={
                "llm_type": "openai",
                "api_key": "synthetic-rejected-key",
                "api_base": "http://rejected.invalid/v1",
                "language_model": "rejected-model",
            },
        ).status_code
    with pytest.raises(configs_block.gr.Error):
        configs_block.apply_llm_config(
            "chat",
            "synthetic-rejected-key",
            "http://rejected.invalid/v1",
            "rejected-model",
            None,
        )
    return "ui-error"


def save_other_config(client, kind, entry, competing_progress):
    try:
        if kind == "embedding":
            if entry == "http":
                result = client.post(
                    "/config/embedding",
                    json={
                        "llm_type": "openai",
                        "api_key": "synthetic-embedding-key",
                        "api_base": "http://embedding.invalid/v1",
                        "language_model": "accepted-embedding",
                    },
                ).status_code
            else:
                result = configs_block.apply_embedding_config(
                    "synthetic-embedding-key",
                    "http://embedding.invalid/v1",
                    "accepted-embedding",
                )
        elif entry == "http":
            result = client.post(
                "/config/rerank",
                json={
                    "reranker_type": "cohere",
                    "api_key": "synthetic-reranker-key",
                    "reranker_model": "accepted-reranker",
                    "cohere_base_url": "http://reranker.invalid/v1/rerank",
                },
            ).status_code
        else:
            result = configs_block.apply_reranker_config(
                "synthetic-reranker-key",
                "accepted-reranker",
                "http://reranker.invalid/v1/rerank",
            )
        assert result == (201 if entry == "http" else 200)
        return result
    finally:
        # Old unprotected writers finish their actual save before releasing the
        # failed LLM. Fixed writers instead signal while blocked in ObservedLock.
        competing_progress.set()


def install_lock(monkeypatch, competing_progress):
    lock = ObservedLock(competing_progress)
    for module in (config, rag_api, configs_block):
        monkeypatch.setattr(module, "llm_config_lock", lock)


def assert_disk_and_memory(settings, env_file, before, kind):
    disk = dotenv_values(env_file)
    llm_fields = rag_api._LLM_CONFIG_FIELDS
    for field in llm_fields:
        assert getattr(settings, field) == before[field]
        expected = before[field]
        assert disk[field.upper()] == (str(expected) if expected is not None else ""), field
    other_field = "openai_embedding_model" if kind == "embedding" else "reranker_model"
    assert getattr(settings, other_field) == f"accepted-{kind}"
    assert disk[other_field.upper()] == f"accepted-{kind}"


@pytest.mark.parametrize("kind", ["embedding", "reranker"])
@pytest.mark.parametrize("save_entry", ["http", "ui"])
@pytest.mark.parametrize("fail_entry", ["http", "ui"])
@pytest.mark.parametrize("failure", ["status", "exception"])
def test_other_config_save_cannot_persist_rejected_llm(
    isolated_settings,
    monkeypatch,
    kind,
    save_entry,
    fail_entry,
    failure,
):
    settings, env_file = isolated_settings
    if fail_entry == "http":
        for role in ROLES:
            setattr(settings, f"{role}_llm_type", "ollama/local")
    seed_file(settings, env_file)
    before = settings.model_dump()
    probe_entered, release_probe, competing_progress = Event(), Event(), Event()
    install_lock(monkeypatch, competing_progress)

    def probe(url, **_kwargs):
        if url.endswith("/chat/completions"):
            probe_entered.set()
            assert release_probe.wait(timeout=10)
            if failure == "exception":
                raise requests.exceptions.ConnectionError("synthetic rejected connection")
            return Mock(status_code=503, text="synthetic rejected response")
        assert url.endswith("/embeddings") or url.endswith("/check-api-key")
        return Mock(status_code=200, text="ok")

    monkeypatch.setattr(configs_block.requests, "post", probe)
    monkeypatch.setattr(configs_block.requests, "get", probe)
    with make_client() as client, ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(fail_llm, client, fail_entry)
        try:
            assert probe_entered.wait(timeout=10)
            saved = pool.submit(save_other_config, client, kind, save_entry, competing_progress)
            assert competing_progress.wait(timeout=10)
        finally:
            release_probe.set()
        expected = (503 if failure == "status" else 400) if fail_entry == "http" else "ui-error"
        assert failed.result(timeout=10) == expected
        assert saved.result(timeout=10) == (201 if save_entry == "http" else 200)
    assert_disk_and_memory(settings, env_file, before, kind)


@pytest.mark.parametrize("kind", ["embedding", "reranker"])
@pytest.mark.parametrize("save_entry", ["http", "ui"])
def test_other_save_waits_through_outer_api_provider_rollback(isolated_settings, monkeypatch, kind, save_entry):
    settings, env_file = isolated_settings
    for role in ROLES:
        setattr(settings, f"{role}_llm_type", "ollama/local")
    seed_file(settings, env_file)
    before = settings.model_dump()
    rollback_entered, release_rollback, competing_progress = Event(), Event(), Event()
    install_lock(monkeypatch, competing_progress)
    original_setattr = LLMConfig.__setattr__

    def pause_provider_rollback(self, field, value):
        if self is settings and field == "chat_llm_type" and value == "ollama/local":
            rollback_entered.set()
            assert release_rollback.wait(timeout=10)
        original_setattr(self, field, value)

    monkeypatch.setattr(LLMConfig, "__setattr__", pause_provider_rollback)

    def probe(url, **_kwargs):
        if url.endswith("/chat/completions"):
            return Mock(status_code=503, text="synthetic rejected response")
        assert url.endswith("/embeddings") or url.endswith("/check-api-key")
        return Mock(status_code=200, text="ok")

    monkeypatch.setattr(configs_block.requests, "post", probe)
    monkeypatch.setattr(configs_block.requests, "get", probe)
    with make_client() as client, ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(fail_llm, client, "http")
        try:
            assert rollback_entered.wait(timeout=10)
            saved = pool.submit(save_other_config, client, kind, save_entry, competing_progress)
            assert competing_progress.wait(timeout=10)
        finally:
            release_rollback.set()
        assert failed.result(timeout=10) == 503
        assert saved.result(timeout=10) == (201 if save_entry == "http" else 200)
    assert_disk_and_memory(settings, env_file, before, kind)
