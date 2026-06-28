"""Shared pytest fixtures.

Builds a Context backed by the stub LLM so tests never need a network or
Ollama. Mirrors what app/main.py assembles, but kept minimal.
"""

import pytest

from app.config import Config
from app.main import build_context


@pytest.fixture
def config() -> Config:
    cfg = Config.load()
    cfg.use_stub_llm = True  # force deterministic, offline behaviour
    return cfg


@pytest.fixture
def ctx(config):
    return build_context(config)
