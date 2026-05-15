"""
src/models/__init__.py
══════════════════════════════════════════════════════════════════════
Models Package Initialization.

This file makes `src/models/` a Python package, enabling clean imports
from other parts of the project.

PACKAGE CONTENTS:
    query_tower            — User encoder (QueryTower class)
    candidate_tower        — Item encoder (CandidateTower class)
    retrieval_model        — Full Two-Tower model (TwoTowerRetrievalModel)
    train_model            — End-to-end training pipeline
    generate_recommendations — Inference and recommendation generation

USAGE FROM OTHER MODULES:
    from src.models import QueryTower, CandidateTower
    from src.models import build_retrieval_model
    from src.models import RetrievalIndex

WHY DO WE NEED __init__.py?
    Python treats directories as packages ONLY when __init__.py exists.
    Without it, you cannot do `from src.models import ...` — Python
    would not know that `models/` is a module.

    Best practice: keep __init__.py lean. Import only the most commonly
    used classes/functions so users can write short import statements.

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

# ── Public API ────────────────────────────────────────────────────────
# These are the symbols exported when someone does `from src.models import *`
# Only expose what external modules need; internals stay internal.

from .query_tower     import QueryTower,               build_query_tower
from .candidate_tower import CandidateTower,           build_candidate_tower
from .retrieval_model import TwoTowerRetrievalModel,   build_retrieval_model

# Package metadata
__version__ = "1.0.0"
__author__  = "Context-Aware Recommendation Engine Team"

__all__ = [
    # Classes
    "QueryTower",
    "CandidateTower",
    "TwoTowerRetrievalModel",

    # Factory functions
    "build_query_tower",
    "build_candidate_tower",
    "build_retrieval_model",
]
