"""
build_interactions.py
=====================
Production script to generate INTERACTION FEATURES for the
Context-Aware Neural Recommendation Engine.

Run from PROJECT ROOT:
    python src/feature_engineering/build_interactions.py

WHY PATH HANDLING MATTERS:
    Python resolves relative paths from your CURRENT WORKING DIRECTORY —
    the folder you are in when you execute the command.

    Because you always run scripts from the project root, all file paths
    must be written relative to THAT folder, not relative to this script.

    WRONG : ../data/processed/   (navigates UP out of src/feature_engineering/)
    RIGHT : data/processed/      (correct relative to project root)

    Using os.path.join() instead of hardcoded slashes also makes the code
    work on both Windows (backslashes) and Linux/macOS (forward slashes).
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────
PROCESSED_DIR = "data/processed"

INPUT_TRANSACTIONS       = os.path.join(PROCESSED_DIR, "transactions_cleaned.csv")
OUTPUT_INTERACTION_FEATURES = os.path.join(PROCESSED_DIR, "interaction_features.csv")


def load_transactions() -> pd.DataFrame:
    """Load the cleaned transactions dataset."""
    print("📂 Loading transactions dataset...")

    transactions = pd.read_csv(INPUT_TRANSACTIONS, parse_dates=["t_dat"])

    print(f"   ✅ transactions : {transactions.shape}")
    return transactions


def build_base_interactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the core customer ↔ article interaction table.

    WHY — Interaction data is the HEART of recommendation systems:
        Every purchase is an implicit positive signal: the user liked this item
        enough to buy it. The Two-Tower model is trained on (user, item) pairs,
        learning embeddings that place similar users and items close together
        in vector space. Without interaction data there is nothing to learn from.
    """
    print("\n🔧 Building base interaction table...")

    interactions = transactions[[
        "customer_id",
        "article_id",
        "t_dat",
        "price",
    ]].copy()

    # ── Implicit feedback label ────────────────────────────────────────────
    # WHY: We treat every purchase as a positive interaction (label = 1).
    # Negative samples (items NOT bought) will be generated during model
    # training via negative sampling — not stored here.
    interactions["label"] = 1

    print(f"   ✅ base interactions shape : {interactions.shape}")
    return interactions


def add_recency_weight(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Add a recency weight so that recent purchases count more.

    WHY — Recency matters in fashion recommendations:
        A dress bought 2 years ago tells us less about current taste than one
        bought last month. Exponential decay is a standard technique to
        down-weight old interactions without throwing them away entirely.

        weight = exp(-λ × days_since_purchase),  λ = 0.01 (soft decay)
    """
    print("\n🔧 Adding recency weights...")

    reference_date = interactions["t_dat"].max()
    interactions["days_since_purchase"] = (
        reference_date - interactions["t_dat"]
    ).dt.days

    DECAY_RATE = 0.01   # lower = slower decay; tune this as a hyperparameter
    interactions["recency_weight"] = np.exp(
        -DECAY_RATE * interactions["days_since_purchase"]
    )

    print(f"   ✅ recency weights added. Sample range: "
          f"{interactions['recency_weight'].min():.4f} – "
          f"{interactions['recency_weight'].max():.4f}")
    return interactions


def add_repeat_purchase_flag(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Flag interactions where the same user bought the same item more than once.

    WHY: Repeat purchases are STRONG positive signals. If a user re-buys an
    item (e.g. the same socks every season), the model should heavily
    reinforce the (user, item) pair. This flag lets the model treat repeats
    as higher-confidence positives.
    """
    print("\n🔧 Adding repeat-purchase flag...")

    purchase_counts = (
        interactions
        .groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="purchase_count_pair")
    )

    interactions = interactions.merge(
        purchase_counts, on=["customer_id", "article_id"], how="left"
    )
    interactions["is_repeat_purchase"] = (
        interactions["purchase_count_pair"] > 1
    ).astype(int)

    print(f"   ✅ repeat-purchase flag added. "
          f"Repeat rows: {interactions['is_repeat_purchase'].sum():,}")
    return interactions


def add_time_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the purchase timestamp.

    WHY — Temporal context improves recommendation quality:
        • day_of_week  → weekend shoppers behave differently from weekday ones
        • month        → seasonal trends (summer dresses, winter coats)
        • week_of_year → promotional periods, holidays

    These features let the model learn context-aware recommendations, which
    is exactly what "Context-Aware" in the project name refers to.
    """
    print("\n🔧 Adding time-based context features...")

    interactions["day_of_week"]  = interactions["t_dat"].dt.dayofweek   # 0=Mon
    interactions["month"]        = interactions["t_dat"].dt.month
    interactions["week_of_year"] = interactions["t_dat"].dt.isocalendar().week.astype(int)
    interactions["year"]         = interactions["t_dat"].dt.year

    print("   ✅ Time features added : day_of_week, month, week_of_year, year")
    return interactions


def save_interactions(interactions: pd.DataFrame) -> None:
    """Save the final interaction feature table to the processed directory."""
    print(f"\n💾 Saving interaction features → {OUTPUT_INTERACTION_FEATURES}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)   # safely create dir if missing
    interactions.to_csv(OUTPUT_INTERACTION_FEATURES, index=False)
    print("   ✅ Saved successfully.")


def main() -> None:
    print("=" * 55)
    print("  INTERACTION FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    transactions = load_transactions()
    interactions = build_base_interactions(transactions)
    interactions = add_recency_weight(interactions)
    interactions = add_repeat_purchase_flag(interactions)
    interactions = add_time_features(interactions)
    save_interactions(interactions)

    print("\n🎯 Interaction feature engineering complete.")
    print(f"   Output : {OUTPUT_INTERACTION_FEATURES}")
    print(f"   Shape  : {interactions.shape}")
    print("=" * 55)


if __name__ == "__main__":
    main()
