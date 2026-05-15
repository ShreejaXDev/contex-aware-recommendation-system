"""
train_model.py
══════════════════════════════════════════════════════════════════════
Training Pipeline for the Two-Tower Retrieval Model.

WHAT DOES THIS SCRIPT DO?
    1. Loads processed feature datasets from data/processed/
    2. Prepares TensorFlow Datasets (train/test split)
    3. Builds the Two-Tower model (Query + Candidate towers)
    4. Trains the model and logs metrics each epoch
    5. Saves the trained model weights to disk
    6. Prints a training summary report

WHY HAVE A SEPARATE TRAINING SCRIPT?
    Notebooks (04_two_tower_model.ipynb) are great for exploration,
    but production ML teams use separate scripts for:

    ✅ Reproducibility  — Run the same training with one command
    ✅ Automation       — Schedule via cron, Airflow, GitHub Actions
    ✅ CI/CD            — Run training as part of deployment pipelines
    ✅ Logging          — Capture metrics to MLflow, W&B, or CSV files
    ✅ Parameterization — Change hyperparams via CLI or config file

HOW TO RUN:
    # From project root:
    python src/models/train_model.py

    # With custom hyperparameters:
    python src/models/train_model.py --embedding_dim 64 --epochs 10

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

import os
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Import our custom modules
from query_tower     import build_query_tower
from candidate_tower import build_candidate_tower
from retrieval_model import build_retrieval_model

# ─────────────────────────────────────────────────────────────────────
# SUPPRESS NOISE (common in TF projects)
# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Default Hyperparameters
# These can be overridden via CLI arguments (see parse_args below)
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data paths
    "data_dir"          : os.path.join("data", "processed"),
    "model_save_dir"    : os.path.join("models", "saved"),

    # Column names (update if your CSVs use different names)
    "user_id_col"       : "user_id",
    "item_id_col"       : "item_id",

    # Model hyperparameters
    "embedding_dim"     : 32,        # Size of embedding vectors
    "use_dense_layers"  : False,     # Whether to add Dense layers in towers
    "dense_units"       : [],        # Dense layer sizes if use_dense_layers=True

    # Training hyperparameters
    "batch_size"        : 128,
    "epochs"            : 5,
    "learning_rate"     : 0.1,
    "train_split"       : 0.8,       # 80% train, 20% test
    "random_seed"       : 42,
}


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_datasets(config: dict) -> tuple:
    """
    Load the three processed CSV datasets from disk.

    Args:
        config (dict): Configuration dictionary with data paths and column names.

    Returns:
        Tuple of (user_df, item_df, interaction_df) as pandas DataFrames.

    Raises:
        FileNotFoundError: If any CSV file is missing.
    """
    data_dir = Path(config["data_dir"])

    paths = {
        "users"        : data_dir / "user_features.csv",
        "items"        : data_dir / "item_features.csv",
        "interactions" : data_dir / "interaction_features.csv",
    }

    # Validate all files exist before loading
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}\n"
                f"Make sure preprocessing pipeline has been run first."
            )

    print("📂 Loading datasets...")
    user_df        = pd.read_csv(paths["users"])
    item_df        = pd.read_csv(paths["items"])
    interaction_df = pd.read_csv(paths["interactions"])

    # -----------------------------------------------------------------
    # FAST EXPERIMENTATION MODE
    # Recommendation datasets are usually huge (millions of interactions).
    # Training on the full data is great for production, but it is often
    # slow on local machines during development.
    # Industry teams commonly prototype on a smaller sampled subset first
    # to validate pipeline correctness, model wiring, and experiment setup.
    # This reduces iteration time from hours to minutes.
    # -----------------------------------------------------------------
    interaction_df = interaction_df.sample(20000, random_state=42)

    # Ensure ID columns are present. H&M-style pipelines often use
    # customer_id/article_id instead of user_id/item_id.
    uid_col = config["user_id_col"]
    iid_col = config["item_id_col"]

    id_cols_missing = (
        uid_col not in user_df.columns
        or uid_col not in interaction_df.columns
        or iid_col not in item_df.columns
        or iid_col not in interaction_df.columns
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
                and fallback_uid in interaction_df.columns
                and fallback_iid in item_df.columns
                and fallback_iid in interaction_df.columns
            ):
                resolved = (fallback_uid, fallback_iid)
                break

        if resolved is None:
            raise KeyError(
                "Could not resolve ID columns across user/item/interaction data. "
                f"Configured: ({uid_col}, {iid_col}). "
                "Expected either (user_id, item_id) or (customer_id, article_id)."
            )

        uid_col, iid_col = resolved
        config["user_id_col"] = uid_col
        config["item_id_col"] = iid_col
        print(
            "   ℹ️  Auto-detected ID columns "
            f"for this run: user='{uid_col}', item='{iid_col}'"
        )

    user_df[uid_col]        = user_df[uid_col].astype(str)
    item_df[iid_col]        = item_df[iid_col].astype(str)
    interaction_df[uid_col] = interaction_df[uid_col].astype(str)
    interaction_df[iid_col] = interaction_df[iid_col].astype(str)

    print(f"   ✅ Users        : {len(user_df):,} rows")
    print(f"   ✅ Items        : {len(item_df):,} rows")
    print(f"   ✅ Interactions (sampled): {len(interaction_df):,} rows")
    print(f"   📐 Sampled interaction_df shape: {interaction_df.shape}")

    return user_df, item_df, interaction_df


# ─────────────────────────────────────────────────────────────────────
# DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────

def prepare_tf_datasets(
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Convert pandas DataFrames into TensorFlow Datasets for training.

    Args:
        user_df        : User features DataFrame
        item_df        : Item features DataFrame
        interaction_df : Interaction features DataFrame
        config         : Configuration dictionary

    Returns:
        Tuple of (train_ds, test_ds, items_ds):
            train_ds  — Batched/cached training interactions
            test_ds   — Batched/cached test interactions
            items_ds  — Dataset of all item IDs (for retrieval index)
    """
    uid_col   = config["user_id_col"]
    iid_col   = config["item_id_col"]
    batch     = config["batch_size"]
    seed      = config["random_seed"]
    split     = config["train_split"]

    # Build interaction dataset from dict of tensors
    interactions_ds = tf.data.Dataset.from_tensor_slices({
        "user_id": tf.constant(interaction_df[uid_col].values),
        "item_id": tf.constant(interaction_df[iid_col].values),
    })

    total_size = len(interaction_df)
    train_size = int(total_size * split)

    # Shuffle first, then split (prevents temporal bias)
    interactions_ds = interactions_ds.shuffle(
        buffer_size=total_size,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    # Batch and cache both splits for efficient training
    train_ds = interactions_ds.take(train_size).batch(batch).cache()
    test_ds  = interactions_ds.skip(train_size).batch(batch).cache()

    # All item IDs dataset for building retrieval index
    items_ds = (
        tf.data.Dataset
        .from_tensor_slices(tf.constant(item_df[iid_col].values))
        .batch(batch)
    )

    print(f"\n⚡ TF Datasets prepared:")
    print(f"   Training batches : {len(train_ds)}  ({train_size:,} interactions)")
    print(f"   Test batches     : {len(test_ds)}  ({total_size - train_size:,} interactions)")
    print(f"   Candidate items  : {len(item_df):,}")

    return train_ds, test_ds, items_ds


# ─────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────

def build_model(
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    items_ds: tf.data.Dataset,
    config: dict,
) -> tf.keras.Model:
    """
    Build the full Two-Tower retrieval model.

    Args:
        user_df  : User features DataFrame (for vocabulary)
        item_df  : Item features DataFrame (for vocabulary)
        items_ds : Dataset of all item IDs (for retrieval evaluation)
        config   : Configuration dictionary

    Returns:
        Compiled TwoTowerRetrievalModel ready for training.
    """
    uid_col = config["user_id_col"]
    iid_col = config["item_id_col"]
    emb_dim = config["embedding_dim"]
    lr      = config["learning_rate"]

    print(f"\n🏗️  Building model (embedding_dim={emb_dim})...")

    # Build user encoder
    query_tower = build_query_tower(
        user_ids=user_df[uid_col].unique().tolist(),
        embedding_dim=emb_dim,
        use_dense_layers=config["use_dense_layers"],
        dense_units=config["dense_units"],
    )

    # Build item encoder
    candidate_tower = build_candidate_tower(
        item_ids=item_df[iid_col].unique().tolist(),
        embedding_dim=emb_dim,
        use_dense_layers=config["use_dense_layers"],
        dense_units=config["dense_units"],
    )

    # Combine into full retrieval model
    model = build_retrieval_model(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        items_dataset=items_ds,
    )

    # Compile with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    )

    print("   ✅ QueryTower     built")
    print("   ✅ CandidateTower built")
    print("   ✅ Retrieval Task configured")
    print(f"   ✅ Compiled with Adam (lr={lr})")

    return model


# ─────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────

def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    config: dict,
) -> dict:
    """
    Train the model and return training history.

    Args:
        model    : Compiled TwoTowerRetrievalModel
        train_ds : Training dataset
        test_ds  : Validation dataset
        config   : Configuration dictionary

    Returns:
        dict: Training history with loss and metric values per epoch.
    """
    epochs = config["epochs"]

    print(f"\n🏋️  Training for {epochs} epoch(s)...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        verbose=1,
    )

    print("=" * 60)
    print("✅ Training complete!\n")

    return history.history


# ─────────────────────────────────────────────────────────────────────
# MODEL SAVING
# ─────────────────────────────────────────────────────────────────────

def save_model(model: tf.keras.Model, config: dict) -> str:
    """
    Save the trained model weights to disk.

    We save only the sub-models (query_tower and candidate_tower)
    because they are the components used at inference time.

    Args:
        model  : Trained TwoTowerRetrievalModel
        config : Configuration dictionary

    Returns:
        str: Path to the saved model directory.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir   = Path(config["model_save_dir"]) / f"two_tower_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save query tower (needed at serving time for user embeddings)
    query_path = save_dir / "query_tower"
    model.query_tower.save_weights(str(query_path / "weights"))

    # Save candidate tower (needed to pre-compute item embeddings)
    candidate_path = save_dir / "candidate_tower"
    model.candidate_tower.save_weights(str(candidate_path / "weights"))

    print(f"💾 Model saved to: {save_dir}")
    return str(save_dir)


# ─────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────

def print_training_report(history: dict, config: dict) -> None:
    """
    Print a human-readable training summary.

    Args:
        history : Training history dict from model.fit()
        config  : Configuration dictionary
    """
    print("\n" + "═" * 60)
    print("📊  TRAINING REPORT")
    print("═" * 60)

    print(f"\n⚙️  Configuration:")
    print(f"   Embedding Dim  : {config['embedding_dim']}")
    print(f"   Batch Size     : {config['batch_size']}")
    print(f"   Epochs         : {config['epochs']}")
    print(f"   Learning Rate  : {config['learning_rate']}")

    print(f"\n📈 Final Epoch Metrics:")
    for key, values in history.items():
        final_val = values[-1]
        print(f"   {key:<52}: {final_val:.4f}")

    # Highlight the most important metric
    top10_key = "val_factorized_top_k/top_10_categorical_accuracy"
    if top10_key in history:
        top10 = history[top10_key][-1]
        print(f"\n🎯 Key Result: Top-10 Accuracy = {top10:.4f}")
        print(f"   Interpretation: For {top10*100:.1f}% of test users, the item")
        print(f"   they actually interacted with appears in our Top-10 recommendations.")

    print("\n" + "═" * 60)


# ─────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments to override default config."""
    parser = argparse.ArgumentParser(
        description="Train the Two-Tower Retrieval Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--embedding_dim",   type=int,   default=DEFAULT_CONFIG["embedding_dim"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--learning_rate",   type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--data_dir",        type=str,   default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--model_save_dir",  type=str,   default=DEFAULT_CONFIG["model_save_dir"])
    parser.add_argument("--user_id_col",     type=str,   default=DEFAULT_CONFIG["user_id_col"])
    parser.add_argument("--item_id_col",     type=str,   default=DEFAULT_CONFIG["item_id_col"])

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────

def main():
    """Main training pipeline — orchestrates all steps end-to-end."""

    print("=" * 60)
    print("🚀 TWO-TOWER RETRIEVAL MODEL — TRAINING PIPELINE")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 0: Parse arguments & build config ──────────────────
    args   = parse_args()
    config = {**DEFAULT_CONFIG, **vars(args)}

    # Set random seeds for reproducibility
    tf.random.set_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # ── Step 1: Load data ────────────────────────────────────────
    user_df, item_df, interaction_df = load_datasets(config)

    # ── Step 2: Prepare TF datasets ─────────────────────────────
    train_ds, test_ds, items_ds = prepare_tf_datasets(
        user_df, item_df, interaction_df, config
    )

    # ── Step 3: Build model ──────────────────────────────────────
    model = build_model(user_df, item_df, items_ds, config)

    # ── Step 4: Train ────────────────────────────────────────────
    history = train_model(model, train_ds, test_ds, config)

    # ── Step 5: Save ─────────────────────────────────────────────
    save_path = save_model(model, config)

    # ── Step 6: Report ───────────────────────────────────────────
    print_training_report(history, config)

    print(f"\n✅ Pipeline complete. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
