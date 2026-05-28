"""
redis_cache.py
====================================================================
Lightweight cache layer using fakeredis for local development.

This module provides small helpers to:
- Build cache keys for recommendations
- Read/write cached responses with expiration
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import fakeredis

logger = logging.getLogger(__name__)

_cache_client: Optional[fakeredis.FakeStrictRedis] = None


def get_cache_client() -> fakeredis.FakeStrictRedis:
    """Return a singleton fakeredis client."""
    global _cache_client
    if _cache_client is None:
        _cache_client = fakeredis.FakeStrictRedis(decode_responses=True)
    return _cache_client


def build_recommendation_key(user_id: str, top_k: int) -> str:
    """Build a stable cache key for user recommendations."""
    return f"recommendations:{user_id}:top{top_k}"


def get_cached_recommendations(user_id: str, top_k: int) -> Optional[dict]:
    """Return cached recommendation payload or None if missing/invalid."""
    key = build_recommendation_key(user_id, top_k)
    try:
        raw = get_cache_client().get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"Cache read failed for key '{key}': {exc}")
        return None


def set_cached_recommendations(
    user_id: str,
    top_k: int,
    payload: dict,
    ttl_seconds: int = 3600,
) -> None:
    """Store recommendation payload with expiration."""
    key = build_recommendation_key(user_id, top_k)
    try:
        serialized = json.dumps(payload)
        get_cache_client().setex(key, ttl_seconds, serialized)
    except Exception as exc:
        logger.warning(f"Cache write failed for key '{key}': {exc}")
