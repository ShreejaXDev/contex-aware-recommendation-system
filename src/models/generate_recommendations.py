"""
generate_recommendations.py
══════════════════════════════════════════════════════════════════════
Recommendation Generation using the trained Two-Tower Retrieval Model.

WHAT DOES THIS SCRIPT DO?
    After training is complete, this script:
    1. Loads the trained model weights from disk
    2. Rebuilds the retrieval index (FAISS ANN nearest-neighbor search)
    3. Generates Top-K item recommendations for any user
    4. Outputs recommendations in a structured format (DataFrame / CSV)

TWO MODES:
    Interactive Mode  — Get recommendations for a single user (live query)
    Batch Mode        — Generate recommendations for all users, export to CSV

WHY SEPARATE FROM TRAINING?
    In production, training and serving are always separate:
    - Training runs once (or on a schedule, e.g. weekly)
    - Serving runs continuously (millions of requests per day)

    This script represents the "offline scoring" part of serving:
    Pre-compute Top-K for all users → store in a fast database (Redis)
    → serve from cache with <10ms latency.

HOW TO RUN:
    # Single user:
    python src/models/generate_recommendations.py --user_id user_001 --top_k 10

    # All users (batch mode):
    python src/models/generate_recommendations.py --batch_mode --output_path recs.csv

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import tensorflow as tf

# Import our custom modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from query_tower     import build_query_tower
from candidate_tower import build_candidate_tower
from src.retrieval.faiss_index import FaissRetrievalIndex, compare_retrieval_performance

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ─────────────────────────────────────────────────────────────────────
# NOTE: Retrieval index uses FAISS ANN for fast vector search.
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# DISPLAY UTILITIES
# ─────────────────────────────────────────────────────────────────────

def print_recommendations(recs_df: pd.DataFrame, user_id: str) -> None:
    """Pretty-print readable recommendations for a single user."""
    print(f"\n🎯 Recommended Products for User {user_id}")
    print("=" * 60)

    for _, row in recs_df.iterrows():
        print(f"\n{int(row['rank'])}.")
        print(f"Product ID: {row['product_id']}")
        print(f"Type: {row['product_type_name']}")
        print(f"Color: {row['colour_group_name']}")
        print(f"Garment Group: {row['garment_group_name']}")
        print(f"Product Group: {row['product_group_name']}")

        detail_desc = row.get("detail_desc", "")
        if pd.notna(detail_desc) and str(detail_desc).strip():
            print(f"Description: {detail_desc}")

        print(f"Score: {row['score']:.4f}")


def load_article_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load article metadata used to turn raw recommendation IDs into readable product details.

    Recommendation models work with IDs internally because IDs are compact, stable tokens
    for embedding lookup and retrieval. Humans usually want product names and attributes,
    so we map the model's article IDs back to metadata after scoring.
    """
    articles_path = data_dir / "articles_cleaned.csv"
    if not articles_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {articles_path}\n"
            "Run preprocessing first so article metadata is available."
        )

    articles_df = pd.read_csv(articles_path)
    if "article_id" not in articles_df.columns:
        raise KeyError("articles_cleaned.csv must contain an 'article_id' column.")

    articles_df["article_id"] = articles_df["article_id"].astype(str)
    return articles_df

def load_trained_towers(
    query_tower,
    candidate_tower,
    saved_models_dir: Path,
) -> None:
    """
    Load trained weights into freshly built tower architectures.

    Inference does not re-train the model. It reuses the learned weights that
    were saved after training so the embeddings reflect real user/item signals
    instead of random initialization.
    """
    query_weights_path = saved_models_dir / "query_tower.weights.h5"
    candidate_weights_path = saved_models_dir / "candidate_tower.weights.h5"

    missing_files = [
        str(path)
        for path in [query_weights_path, candidate_weights_path]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Trained model weights not found.\n"
            f"Missing files: {missing_files}\n"
            "Run 'python src/models/train_model.py' first to save learned weights into saved_models/."
        )

    # Keras models need to be built before variables can be restored.
    # A single dummy forward pass creates the embedding weights with the
    # same shapes used during training.
    query_tower(tf.constant([query_tower.user_lookup.get_vocabulary()[0]]))
    candidate_tower(tf.constant([candidate_tower.item_lookup.get_vocabulary()[0]]))

    query_tower.load_weights(str(query_weights_path), by_name=True, skip_mismatch=True)
    candidate_tower.load_weights(str(candidate_weights_path), by_name=True, skip_mismatch=True)

    print("   ✅ Loaded trained weights from saved_models/")


def build_readable_recommendations(
    recs_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    item_id_col: str,
) -> pd.DataFrame:
    """
    Join model recommendations with article metadata.

    This is the postprocessing step in a recommendation pipeline:
    - the model outputs ranked article IDs
    - we join those IDs to metadata
    - we rename columns into a friendlier display format
    """
    readable_df = recs_df.merge(
        articles_df,
        left_on=item_id_col,
        right_on="article_id",
        how="left",
    )

    readable_df = readable_df.rename(
        columns={
            item_id_col: "product_id",
        }
    )

    desired_columns = [
        "user_id",
        "rank",
        "product_id",
        "product_type_name",
        "colour_group_name",
        "garment_group_name",
        "product_group_name",
        "detail_desc",
        "score",
    ]

    for column in desired_columns:
        if column not in readable_df.columns:
            readable_df[column] = "Unknown"

    readable_df = readable_df[desired_columns]
    readable_df = readable_df.fillna("Unknown")

    return readable_df


def load_model_config(saved_models_dir: Path) -> dict:
    """
    Load the architecture settings saved during training.

    This keeps inference aligned with training even if the embedding size or
    optional dense layers were changed in a past experiment.
    """
    config_path = saved_models_dir / "model_config.json"
    if not config_path.exists():
        return {
            "embedding_dim": 32,
            "use_dense_layers": False,
            "dense_units": [],
        }

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate recommendations using the trained Two-Tower model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        help="Single user ID to generate recommendations for",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of recommendations to generate",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Generate recommendations for ALL users",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="recommendations.csv",
        help="Output CSV path for batch mode",
    )
    parser.add_argument(
        "--user_id_col",
        type=str,
        default="user_id",
        help="User ID column name in CSVs",
    )
    parser.add_argument(
        "--item_id_col",
        type=str,
        default="item_id",
        help="Item ID column name in CSVs",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Embedding dimension (must match trained model)",
    )
    parser.add_argument(
        "--compare_retrieval",
        action="store_true",
        help="Print BruteForce vs FAISS retrieval speed comparison",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────

def main():
    """Main recommendation generation pipeline."""
    print("=" * 60)
    print("🎯 TWO-TOWER RETRIEVAL — RECOMMENDATION GENERATION")
    print("=" * 60)

    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    saved_models_dir = Path("saved_models")
    model_config = load_model_config(saved_models_dir)
    uid_col  = args.user_id_col
    iid_col  = args.item_id_col

    print("\n📂 Loading data...")
    user_df = pd.read_csv(data_dir / "user_features.csv")
    item_df = pd.read_csv(data_dir / "item_features.csv")
    articles_df = load_article_metadata(data_dir)

    id_cols_missing = (
        uid_col not in user_df.columns
        or iid_col not in item_df.columns
    )

    if id_cols_missing:
        fallback_pairs = [
            ("customer_id", "article_id"),
            ("user_id", "item_id"),
        ]

        resolved = None
        for fallback_uid, fallback_iid in fallback_pairs:
            if (
                fallback_uid in user_df.columns
                and fallback_iid in item_df.columns
            ):
                resolved = (fallback_uid, fallback_iid)
                break

        if resolved is None:
            raise KeyError(
                "Could not resolve ID columns across user/item data. "
                f"Configured: ({uid_col}, {iid_col}). "
                "Expected either (user_id, item_id) or (customer_id, article_id)."
            )

        uid_col, iid_col = resolved
        print(
            "   ℹ️  Auto-detected ID columns "
            f"for this run: user='{uid_col}', item='{iid_col}'"
        )

    user_df[uid_col] = user_df[uid_col].astype(str)
    item_df[iid_col] = item_df[iid_col].astype(str)

    all_user_ids = user_df[uid_col].unique().tolist()
    all_item_ids = item_df[iid_col].unique().tolist()

    print(f"   ✅ {len(all_user_ids):,} users | {len(all_item_ids):,} items")

    # ── Build towers and load trained weights ─────────────────────
    # The architecture is created first, then the learned weights are
    # loaded from disk. This is what turns a model definition into a
    # real inference model that uses trained embeddings.
    print("\n🏗️  Initializing model towers...")

    query_tower = build_query_tower(
        user_ids=all_user_ids,
        embedding_dim=args.embedding_dim,
    )
    candidate_tower = build_candidate_tower(
        item_ids=all_item_ids,
        embedding_dim=args.embedding_dim,
    )

    # Load the trained weights before generating any recommendations.
    # Without this step, the towers would use random initialization and the
    # output recommendations would not reflect what the model learned.
    saved_models_dir = Path("saved_models")
    load_trained_towers(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        saved_models_dir=saved_models_dir,
    )

    # ── Build retrieval index ─────────────────────────────────────
    try:
        index = FaissRetrievalIndex(
            query_tower=query_tower,
            candidate_tower=candidate_tower,
            item_ids=all_item_ids,
            top_k=args.top_k,
        )
    except ImportError as exc:
        raise SystemExit(
            "FAISS is not installed. Install it with: pip install faiss-cpu"
        ) from exc

    if args.compare_retrieval:
        comparison = compare_retrieval_performance(
            query_tower=query_tower,
            candidate_tower=candidate_tower,
            item_ids=all_item_ids,
            user_ids=all_user_ids,
            top_k=args.top_k,
        )
        print(comparison)

    # ── Generate recommendations ──────────────────────────────────
    if args.batch_mode:
        # Batch mode: all users
        recs_df = index.recommend_batch(
            user_ids=all_user_ids,
            top_k=args.top_k,
        )

        # Postprocess the raw IDs into readable article information.
        # The model still ranks IDs internally; this merge makes the output useful to humans.
        readable_recs_df = build_readable_recommendations(
            recs_df=recs_df,
            articles_df=articles_df,
            item_id_col="item_id",
        )

        output_path = Path(args.output_path)
        readable_recs_df.to_csv(output_path, index=False)
        print(f"\n💾 Recommendations saved to: {output_path}")

    else:
        # Single user mode
        target_user = args.user_id or all_user_ids[0]
        recs_df = index.recommend(user_id=target_user, top_k=args.top_k)

        # Postprocess the raw article IDs into readable product metadata.
        # This is the final recommendation-serving step: score first, explain second.
        readable_recs_df = build_readable_recommendations(
            recs_df=recs_df,
            articles_df=articles_df,
            item_id_col="item_id",
        )

        print_recommendations(readable_recs_df, target_user)


if __name__ == "__main__":
    main()
