#!/usr/bin/env python3
"""Shared TypeSafe (Jev) client helpers for bridge analysis scripts.

Client: TYPESAFE_API_KEY from the environment; without a key get_client()
returns None and callers must serve strictly from the cache, failing
loudly on a cache miss (never calling out).
Answers are cached to a JSON sidecar keyed by a content hash of
(dataset, question version, state), so a dataset release is judged once
and every later run is free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from typesafe_sdk import TypeSafeClient

# Bump when instructions/criteria change meaningfully so old cache entries
# cannot masquerade as answers to the new question.
QUESTION_VERSION = 1

CACHE_PATH = Path(__file__).parent / "out" / "tsafe_cache.json"

# Pinned to the stable latest Jev; list via get_client().models.list().
MODEL = "jev-latest"


def get_client() -> TypeSafeClient | None:
    """Live client when TYPESAFE_API_KEY is set, else None: callers must
    then serve strictly from the AnswerCache and fail loudly on a miss
    (the SDK has no silent cache-only mode — reviewed 2026-09-17)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return None
    return TypeSafeClient(
        api_key=key,
        base_url=os.environ.get("TYPESAFE_BASE_URL"),
        model=MODEL,
        timeout=60.0,
    )


def cache_key(namespace: str, state) -> str:
    blob = json.dumps(
        {"v": QUESTION_VERSION, "ns": namespace, "state": state},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class AnswerCache:
    """JSON-backed memo for one question namespace."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.data: dict = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))
