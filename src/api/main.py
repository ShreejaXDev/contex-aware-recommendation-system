"""
===============================================================
 Context-Aware Neural Recommendation Engine — FastAPI Service
===============================================================

What this file does:
  - Wraps your trained Two-Tower recommendation model in a REST API
  - Loads model weights ONCE at startup (fast for all future requests)
  - Exposes clean HTTP endpoints any frontend or app can call
  - Returns structured JSON with ranked product recommendations

Key concepts for interns:
  - INFERENCE: running a trained model on new inputs to get predictions
  - SERVING:   deploying that model so other programs can use it via HTTP
  - STARTUP:   loading heavy resources (model, data) before requests arrive
  - ENDPOINT:  a URL that accepts requests and returns responses
"""

import os
import sys
import logging
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Add project root to path so we can import from src/models/
# ---------------------------------------------------------------------------
# This lets Python find query_tower.py, candidate_tower.py, etc.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.query_tower import QueryTower
from src.models.candidate_tower import CandidateTower
from src.models.retrieval_model import RetrievalModel

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# We use logging instead of print() in production code.
# Logging adds timestamps, severity levels, and is easier to filter.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_first_existing_path(base_dir: Path, candidates: list[str]) -> Path:
    """Return the first existing path from a list of candidate filenames."""
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected files were found. "
        f"Checked: {', '.join(str(base_dir / candidate) for candidate in candidates)}"
    )


# ============================================================
# SECTION 1 — PYDANTIC RESPONSE MODELS
# ============================================================
# Pydantic models define the exact shape of our JSON responses.
# FastAPI uses these to auto-generate docs at /docs and validate output.

class RecommendationItem(BaseModel):
    """A single recommended product returned in the response."""
    rank: int                         # Position in the recommendation list (1 = best)
    product_id: str                   # Unique product identifier
    product_type: Optional[str]       # e.g. "Hoodie", "T-shirt"
    color: Optional[str]              # e.g. "Black", "Navy"
    garment_group: Optional[str]      # e.g. "Jersey Basic"
    product_group: Optional[str]      # e.g. "Garment Upper Body"
    score: Optional[float]            # Similarity score (higher = more relevant)


class RecommendationResponse(BaseModel):
    """The full JSON response returned by GET /recommend/{user_id}."""
    user_id: str
    total_recommendations: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    """Response for the root health-check endpoint."""
    message: str
    model_loaded: bool
    status: str


# ============================================================
# SECTION 2 — GLOBAL STATE (loaded once at startup)
# ============================================================
# We store the model and data here so every request reuses them.
# Loading TensorFlow models takes several seconds — doing it per-request
# would make the API unusably slow.

class AppState:
    """Holds all resources that are expensive to load."""
    model: Optional[RetrievalModel] = None           # The full Two-Tower model
    retrieval_index = None                            # BruteForce lookup index
    articles_df: Optional[pd.DataFrame] = None       # Product metadata
    user_ids: Optional[list] = None                  # Known user IDs from training


app_state = AppState()


# ============================================================
# SECTION 3 — STARTUP / SHUTDOWN LIFECYCLE
# ============================================================
# FastAPI's lifespan context manager runs code:
#   - BEFORE the first request arrives  (startup)
#   - AFTER the server shuts down       (shutdown)
#
# This is where we load the model, weights, and dataset.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model weights and product metadata.
    Shutdown: clean up resources.
    """
    # ---- STARTUP ----
    logger.info("=== API starting up — loading model and data ===")

    # ------------------------------------------------------------------
    # Step 1: Load product metadata (articles)
    # ------------------------------------------------------------------
    # This CSV maps product IDs to human-readable names and attributes.
    processed_dir = PROJECT_ROOT / "data" / "processed"
    saved_models_dir = PROJECT_ROOT / "saved_models"

    articles_path = resolve_first_existing_path(
        processed_dir,
        ["articles.csv", "articles_cleaned.csv"],
    )
    app_state.articles_df = pd.read_csv(articles_path)
    if "article_id" in app_state.articles_df.columns:
        app_state.articles_df["article_id"] = app_state.articles_df["article_id"].astype(str)
    elif "item_id" in app_state.articles_df.columns:
        app_state.articles_df = app_state.articles_df.rename(columns={"item_id": "article_id"})
        app_state.articles_df["article_id"] = app_state.articles_df["article_id"].astype(str)
    else:
        raise KeyError(
            f"{articles_path.name} must contain either 'article_id' or 'item_id'."
        )
    logger.info(f"Loaded {len(app_state.articles_df):,} products from {articles_path.name}")

    # ------------------------------------------------------------------
    # Step 2: Load candidate item IDs (the products the model knows about)
    # ------------------------------------------------------------------
    # The candidate tower embeds all possible products.
    # We need these IDs to build the retrieval index.
    items_path = processed_dir / "unique_article_ids.npy"
    if items_path.exists():
        unique_article_ids = np.load(items_path, allow_pickle=True)
        logger.info(f"Loaded {len(unique_article_ids):,} candidate product IDs from unique_article_ids.npy")
    else:
        unique_article_ids = app_state.articles_df["article_id"].astype(str).unique()
        logger.warning(
            "unique_article_ids.npy not found; deriving candidate IDs from the articles metadata instead."
        )
    candidate_ids_dataset = tf.data.Dataset.from_tensor_slices(
        unique_article_ids.astype(str)
    )

    # ------------------------------------------------------------------
    # Step 3: Load user IDs seen during training
    # ------------------------------------------------------------------
    users_path = processed_dir / "unique_customer_ids.npy"
    if users_path.exists():
        app_state.user_ids = set(np.load(users_path, allow_pickle=True).astype(str).tolist())
        logger.info(f"Loaded {len(app_state.user_ids):,} known user IDs from unique_customer_ids.npy")
    else:
        customers_path = resolve_first_existing_path(
            processed_dir,
            ["customers.csv", "customers_cleaned.csv"],
        )
        customers_df = pd.read_csv(customers_path)
        user_id_column = "customer_id" if "customer_id" in customers_df.columns else "user_id"
        if user_id_column not in customers_df.columns:
            raise KeyError(
                f"{customers_path.name} must contain either 'customer_id' or 'user_id'."
            )
        app_state.user_ids = set(customers_df[user_id_column].astype(str).tolist())
        logger.warning(
            "unique_customer_ids.npy not found; deriving known user IDs from the customers metadata instead."
        )

    # ------------------------------------------------------------------
    # Step 4: Build the Two-Tower model and load saved weights
    # ------------------------------------------------------------------
    # We reconstruct the same model architecture used during training,
    # then load the saved weights into it.
    if not saved_models_dir.is_dir():
        raise FileNotFoundError(
            f"Saved model directory '{saved_models_dir}' not found. "
            "Train the model first with: python src/models/train_model.py"
        )

    model_config_path = saved_models_dir / "model_config.json"
    if model_config_path.exists():
        with model_config_path.open("r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    else:
        model_config = {
            "embedding_dim": 32,
            "use_dense_layers": False,
            "dense_units": [],
        }

    # Reconstruct model (same architecture as training)
    query_tower = QueryTower(
        user_vocabulary=tf.keras.layers.StringLookup(
            vocabulary=sorted(app_state.user_ids),
            mask_token=None,
            name="user_vocabulary",
        ),
        embedding_dim=model_config.get("embedding_dim", 32),
        use_dense_layers=model_config.get("use_dense_layers", False),
        dense_units=model_config.get("dense_units", []),
    )
    candidate_tower = CandidateTower(
        item_vocabulary=tf.keras.layers.StringLookup(
            vocabulary=sorted(unique_article_ids.astype(str).tolist()),
            mask_token=None,
            name="item_vocabulary",
        ),
        embedding_dim=model_config.get("embedding_dim", 32),
        use_dense_layers=model_config.get("use_dense_layers", False),
        dense_units=model_config.get("dense_units", []),
    )

    app_state.model = RetrievalModel(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        items_dataset=candidate_ids_dataset,
    )

    query_weights_path = saved_models_dir / "query_tower.weights.h5"
    candidate_weights_path = saved_models_dir / "candidate_tower.weights.h5"
    if not query_weights_path.exists() or not candidate_weights_path.exists():
        raise FileNotFoundError(
            "Missing trained tower weights. Expected files: "
            f"{query_weights_path} and {candidate_weights_path}."
        )

    # Build variables before loading weights.
    query_tower(tf.constant([next(iter(app_state.user_ids))]))
    candidate_tower(tf.constant([str(unique_article_ids[0])]))

    query_tower.load_weights(str(query_weights_path))
    candidate_tower.load_weights(str(candidate_weights_path))
    logger.info("Loaded trained query and candidate tower weights")

    # ------------------------------------------------------------------
    # Step 5: Build the retrieval index (BruteForce)
    # ------------------------------------------------------------------
    # TFRS BruteForce index pre-computes all candidate embeddings so
    # recommendation lookups are fast at inference time.
    #
    # How it works:
    #   1. candidate_tower encodes every product into an embedding vector
    #   2. At query time, user embedding is compared to all product embeddings
    #   3. Top-K most similar products are returned
    logger.info("Building retrieval index (pre-computing candidate embeddings)...")

    app_state.retrieval_index = tfrs.layers.factorized_top_k.BruteForce(
        app_state.model.query_tower
    )
    app_state.retrieval_index.index_from_dataset(
        candidate_ids_dataset.batch(128).map(
            lambda ids: (ids, app_state.model.candidate_tower(ids))
        )
    )
    logger.info("Retrieval index built — API is ready to serve requests!")

    # ---- Hand control to FastAPI (requests are now handled) ----
    yield

    # ---- SHUTDOWN ----
    logger.info("=== API shutting down — cleaning up resources ===")
    app_state.model = None
    app_state.retrieval_index = None
    app_state.articles_df = None


# ============================================================
# SECTION 4 — FASTAPI APP
# ============================================================
# We pass our lifespan function so startup/shutdown hooks fire correctly.

app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description=(
        "Two-Tower recommendation model served via FastAPI. "
        "Returns personalized product recommendations for a given user ID."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# SECTION 5 — ENDPOINTS
# ============================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """
    Health check endpoint.

    Returns the API status and whether the model is loaded.
    Useful for monitoring and deployment checks.
    """
    return HealthResponse(
        message="Recommendation API is running",
        model_loaded=app_state.model is not None,
        status="ok" if app_state.model is not None else "model_not_loaded",
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get personalized recommendations for a user",
    description=(
        "Returns the top N product recommendations for the given user ID. "
        "The model uses a Two-Tower architecture to find products whose "
        "embeddings are closest to the user's embedding."
    ),
)
def recommend(user_id: str, top_k: int = 10):
    """
    Generate personalized product recommendations.

    Parameters
    ----------
    user_id : str
        The customer ID to generate recommendations for.
        Must be a user seen during model training.
    top_k : int, optional
        Number of recommendations to return (default: 10, max: 100).

    Returns
    -------
    RecommendationResponse
        Ranked list of recommended products with metadata and scores.
    """
    # ------------------------------------------------------------------
    # Guard: model must be loaded
    # ------------------------------------------------------------------
    if app_state.model is None or app_state.retrieval_index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. The server may still be starting up. "
                "Please wait a moment and try again."
            ),
        )

    # ------------------------------------------------------------------
    # Guard: cap top_k to a sensible limit
    # ------------------------------------------------------------------
    top_k = max(1, min(top_k, 100))

    # ------------------------------------------------------------------
    # Guard: validate user ID
    # ------------------------------------------------------------------
    # We only warn (not block) for unknown users because you may want to
    # generate cold-start recommendations for new users in the future.
    if app_state.user_ids is not None and user_id not in app_state.user_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"User ID '{user_id}' was not seen during model training. "
                "Recommendations may be unreliable. "
                "Pass a valid customer ID from the training dataset."
            ),
        )

    # ------------------------------------------------------------------
    # INFERENCE: run the retrieval index to get top-K product IDs
    # ------------------------------------------------------------------
    # This is the core of the recommendation pipeline:
    #   1. query_tower encodes the user_id into an embedding vector
    #   2. retrieval_index finds the nearest product embeddings
    #   3. Returns (scores, product_ids) for the top K candidates
    try:
        user_tensor = tf.constant([user_id])  # Model expects a batch dimension
        scores, product_ids = app_state.retrieval_index(user_tensor, k=top_k)

        # Remove batch dimension — we sent one user, we get one result
        scores = scores.numpy()[0]          # shape: (top_k,)
        product_ids = product_ids.numpy()[0]  # shape: (top_k,)

        # Decode bytes → strings if needed (TF sometimes returns bytes)
        product_ids = [
            pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
            for pid in product_ids
        ]

    except Exception as e:
        logger.error(f"Inference failed for user '{user_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}",
        )

    # ------------------------------------------------------------------
    # Guard: empty results
    # ------------------------------------------------------------------
    if not product_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No recommendations could be generated for user '{user_id}'. "
                "The user may not have enough interaction history."
            ),
        )

    # ------------------------------------------------------------------
    # METADATA MAPPING: enrich product IDs with human-readable attributes
    # ------------------------------------------------------------------
    # The model only knows about product IDs (e.g. "673677002").
    # We join against articles.csv to get names, colors, groups, etc.
    articles_df = app_state.articles_df.set_index("article_id")

    recommendations = []
    for rank, (pid, score) in enumerate(zip(product_ids, scores), start=1):
        # Look up this product in the metadata table
        if pid in articles_df.index:
            row = articles_df.loc[pid]
            item = RecommendationItem(
                rank=rank,
                product_id=pid,
                product_type=_safe_str(row.get("product_type_name")),
                color=_safe_str(row.get("colour_group_name")),
                garment_group=_safe_str(row.get("garment_group_name")),
                product_group=_safe_str(row.get("product_group_name")),
                score=round(float(score), 6),
            )
        else:
            # Product ID exists in model but not in metadata (rare edge case)
            logger.warning(f"Product ID {pid} not found in articles metadata")
            item = RecommendationItem(
                rank=rank,
                product_id=pid,
                product_type=None,
                color=None,
                garment_group=None,
                product_group=None,
                score=round(float(score), 6),
            )
        recommendations.append(item)

    return RecommendationResponse(
        user_id=user_id,
        total_recommendations=len(recommendations),
        recommendations=recommendations,
    )


# ============================================================
# SECTION 6 — HELPER FUNCTIONS
# ============================================================

def _safe_str(value) -> Optional[str]:
    """
    Safely convert a pandas value to string, returning None for NaN/null.
    Prevents NaN values from appearing in JSON responses.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


# ============================================================
# SECTION 7 — ENTRY POINT (for direct script execution)
# ============================================================
# This block runs only when you execute: python src/api/main.py
# It is NOT used when running with uvicorn (which imports the module).

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # Auto-restarts when you edit the file (dev only)
        log_level="info",
    )
