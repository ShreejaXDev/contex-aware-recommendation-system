"""
=============================================================
 Recommendation Model Evaluation (Top-K Metrics)
=============================================================

This script evaluates a trained Two-Tower recommendation model using
industry-standard Top-K metrics:

- Recall@K
- Precision@K
- NDCG@K

Why evaluation matters (simple explanation):
- Training teaches the model to learn user/item patterns.
- Evaluation checks whether those patterns actually produce useful
  recommendations on unseen data.
- Top-K metrics focus on the *ranked* list users see, not on raw scores.
- This turns a "working demo" into a scientifically evaluated system.

Evaluation flow:
    Test User
        -> Generate Top-10 Recommendations
        -> Compare with Actual User Interactions
        -> Compute Metrics
        -> Display Final Scores

Run from project root:
    python src/evaluation/evaluate_model.py
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


# ─────────────────────────────────────────────────────────────────────
# PROJECT SETUP
# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.query_tower import build_query_tower
from src.models.candidate_tower import build_candidate_tower
from src.models.retrieval_model import RetrievalModel


# ─────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────
DEFAULT_K = 10
DEFAULT_MAX_USERS = 500
DEFAULT_MIN_INTERACTIONS = 2
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42


# ─────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────

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


def load_model_config(saved_models_dir: Path) -> dict:
    """Load the model architecture settings saved during training."""
    config_path = saved_models_dir / "model_config.json"
    if not config_path.exists():
        return {
            "embedding_dim": 32,
            "use_dense_layers": False,
            "dense_units": [],
        }

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load processed user, item, and interaction datasets with column detection.

    Returns:
        user_df, item_df, interaction_df, user_id_col, item_id_col
    """
    user_path = resolve_first_existing_path(data_dir, ["user_features.csv", "customers_cleaned.csv", "customers.csv"])
    item_path = resolve_first_existing_path(data_dir, ["item_features.csv", "articles_cleaned.csv", "articles.csv"])
    interaction_path = resolve_first_existing_path(data_dir, ["interaction_features.csv", "transactions_cleaned.csv"])

    user_df = pd.read_csv(user_path)
    item_df = pd.read_csv(item_path)
    interaction_df = pd.read_csv(interaction_path)

    # Detect ID column names (H&M style: customer_id/article_id)
    candidate_pairs = [
        ("user_id", "item_id"),
        ("customer_id", "article_id"),
    ]

    resolved = None
    for uid, iid in candidate_pairs:
        if uid in user_df.columns and iid in item_df.columns and uid in interaction_df.columns and iid in interaction_df.columns:
            resolved = (uid, iid)
            break

    if resolved is None:
        raise KeyError(
            "Could not resolve user/item ID columns across datasets. "
            "Expected (user_id, item_id) or (customer_id, article_id)."
        )

    user_id_col, item_id_col = resolved

    user_df[user_id_col] = user_df[user_id_col].astype(str)
    item_df[item_id_col] = item_df[item_id_col].astype(str)
    interaction_df[user_id_col] = interaction_df[user_id_col].astype(str)
    interaction_df[item_id_col] = interaction_df[item_id_col].astype(str)

    return user_df, item_df, interaction_df, user_id_col, item_id_col


def build_model_and_index(
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    saved_models_dir: Path,
    batch_size: int,
) -> tfrs.layers.factorized_top_k.BruteForce:
    """
    Rebuild the Two-Tower model, load trained weights, and build a retrieval index.
    """
    model_config = load_model_config(saved_models_dir)

    query_tower = build_query_tower(
        user_ids=user_df[user_id_col].unique().tolist(),
        embedding_dim=model_config.get("embedding_dim", 32),
        use_dense_layers=model_config.get("use_dense_layers", False),
        dense_units=model_config.get("dense_units", []),
    )
    candidate_tower = build_candidate_tower(
        item_ids=item_df[item_id_col].unique().tolist(),
        embedding_dim=model_config.get("embedding_dim", 32),
        use_dense_layers=model_config.get("use_dense_layers", False),
        dense_units=model_config.get("dense_units", []),
    )

    items_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(item_df[item_id_col].values)
    ).batch(batch_size)

    _ = RetrievalModel(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        items_dataset=items_ds,
    )

    query_weights_path = saved_models_dir / "query_tower.weights.h5"
    candidate_weights_path = saved_models_dir / "candidate_tower.weights.h5"

    if not query_weights_path.exists() or not candidate_weights_path.exists():
        raise FileNotFoundError(
            "Missing trained tower weights. Expected files: "
            f"{query_weights_path} and {candidate_weights_path}."
        )

    # Build variables before loading weights.
    query_tower(tf.constant([user_df[user_id_col].iloc[0]]))
    candidate_tower(tf.constant([item_df[item_id_col].iloc[0]]))

    query_tower.load_weights(str(query_weights_path))
    candidate_tower.load_weights(str(candidate_weights_path))

    # Build retrieval index
    index = tfrs.layers.factorized_top_k.BruteForce(query_tower)
    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(tf.constant(item_df[item_id_col].values)).batch(batch_size).map(
            lambda ids: (ids, candidate_tower(ids))
        )
    )

    return index


def sample_users_for_evaluation(
    interaction_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    max_users: int,
    min_interactions: int,
    seed: int,
) -> list[str]:
    """
    Sample a subset of users for evaluation.

    Why sampling?
    - Full evaluation on the entire dataset can be very slow (millions of rows).
    - Sampling a smaller group of users is common in experimentation to
      validate correctness and iterate faster.
    """
    user_counts = interaction_df.groupby(user_id_col)[item_id_col].nunique()
    eligible_users = user_counts[user_counts >= min_interactions].index.tolist()

    if not eligible_users:
        return []

    if max_users is None or max_users <= 0 or max_users >= len(eligible_users):
        return eligible_users

    rng = np.random.default_rng(seed)
    return rng.choice(eligible_users, size=max_users, replace=False).tolist()


def get_top_k_recommendations(
    index: tfrs.layers.factorized_top_k.BruteForce,
    user_id: str,
    k: int,
) -> list[str]:
    """Return a list of top-K item IDs for a user."""
    scores, item_ids = index(tf.constant([user_id]), k=k)
    if item_ids.shape[1] == 0:
        return []

    return [
        item_id.decode("utf-8") if isinstance(item_id, bytes) else str(item_id)
        for item_id in item_ids[0].numpy()
    ]


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K = (relevant items in top-K) / K

    Simple meaning: Of the K items we recommended, how many were actually relevant?
    """
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K = (relevant items in top-K) / (total relevant items)

    Simple meaning: Of all the items the user actually interacted with,
    how many did we successfully place inside the top-K list?
    """
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)

    Simple meaning:
    - Rewards correct recommendations more when they appear near the top.
    - A hit at rank 1 counts more than a hit at rank 10.
    """
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def build_relevant_items_map(
    interaction_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    users: list[str],
) -> dict:
    """
    Build a map of user_id -> set of interacted item_ids.

    We prefilter to the sampled users to keep evaluation fast
    on large datasets.
    """
    if users:
        filtered = interaction_df[interaction_df[user_id_col].isin(users)]
    else:
        filtered = interaction_df

    grouped = filtered.groupby(user_id_col)[item_id_col].unique()
    return {user_id: set(items.tolist()) for user_id, items in grouped.items()}


def evaluate(
    index: tfrs.layers.factorized_top_k.BruteForce,
    interaction_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    users: list[str],
    k: int,
) -> dict:
    """Compute average Top-K metrics across sampled users."""
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    interactions_grouped = build_relevant_items_map(
        interaction_df=interaction_df,
        user_id_col=user_id_col,
        item_id_col=item_id_col,
        users=users,
    )

    for user_id in users:
        relevant_items = interactions_grouped.get(user_id, set())
        if not relevant_items:
            continue

        recommended_items = get_top_k_recommendations(index, user_id, k)
        if not recommended_items:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue

        precision_scores.append(precision_at_k(recommended_items, relevant_items, k))
        recall_scores.append(recall_at_k(recommended_items, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommended_items, relevant_items, k))

    if not precision_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "users_evaluated": 0,
        }

    return {
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
        "ndcg": float(np.mean(ndcg_scores)),
        "users_evaluated": len(precision_scores),
    }


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Two-Tower recommendation model with Top-K metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"))
    parser.add_argument("--saved_models_dir", type=str, default="saved_models")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max_users", type=int, default=DEFAULT_MAX_USERS)
    parser.add_argument("--min_interactions", type=int, default=DEFAULT_MIN_INTERACTIONS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    saved_models_dir = PROJECT_ROOT / args.saved_models_dir

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    if not saved_models_dir.exists():
        print(f"❌ Saved models directory not found: {saved_models_dir}")
        print("   Run training first to generate saved_models/.")
        return

    try:
        user_df, item_df, interaction_df, user_id_col, item_id_col = load_datasets(data_dir)
    except Exception as exc:
        print(f"❌ Failed to load datasets: {exc}")
        return

    # Sampling is common for large datasets to keep evaluation fast.
    users_to_eval = sample_users_for_evaluation(
        interaction_df,
        user_id_col,
        item_id_col,
        max_users=args.max_users,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )

    if not users_to_eval:
        print("❌ No eligible users found for evaluation.")
        print("   Try lowering --min_interactions or check the interaction dataset.")
        return

    try:
        index = build_model_and_index(
            user_df=user_df,
            item_df=item_df,
            user_id_col=user_id_col,
            item_id_col=item_id_col,
            saved_models_dir=saved_models_dir,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"❌ Failed to build model or load weights: {exc}")
        return

    results = evaluate(
        index=index,
        interaction_df=interaction_df,
        user_id_col=user_id_col,
        item_id_col=item_id_col,
        users=users_to_eval,
        k=args.k,
    )

    print("\n" + "=" * 50)
    print("RECOMMENDATION MODEL EVALUATION")
    print("=" * 50)
    print(f"Users evaluated : {results['users_evaluated']}")
    print(f"Recall@{args.k:<2}       : {results['recall']:.4f}")
    print(f"Precision@{args.k:<2}    : {results['precision']:.4f}")
    print(f"NDCG@{args.k:<2}         : {results['ndcg']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
