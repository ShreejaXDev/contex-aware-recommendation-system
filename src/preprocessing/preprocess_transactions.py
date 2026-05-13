"""
preprocess_transactions.py
==========================
Production preprocessing script for the H&M transactions dataset.

Purpose:
    - Load raw transactions_train.csv
    - Clean and validate all transaction records
    - Convert date column to datetime
    - Remove duplicates
    - Handle missing values
    - Sort chronologically (critical for time-based ML splits)
    - Save cleaned output to data/processed/transactions_cleaned.csv

Why transactions are the most critical dataset:
    Transactions are the INTERACTION SIGNAL in collaborative filtering.
    This is the "who bought what" matrix that drives personalization.
    Every recommendation the model makes is ultimately derived from this data.
    Clean, correctly-typed transaction data is non-negotiable.

Author : Context-Aware Neural Recommendation Engine — Internship Project
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ── Logger Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── Path Constants ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw',       'transactions_train.csv')
OUTPUT_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'transactions_cleaned.csv')


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_raw_transactions(filepath: str) -> pd.DataFrame:
    """
    Load raw transactions CSV from disk.

    Args:
        filepath: Absolute path to transactions_train.csv

    Returns:
        Raw DataFrame (no modifications)

    Performance note:
        The transactions file is large (~3.6M rows). Loading with explicit dtypes
        reduces memory usage significantly by preventing pandas from auto-inferring.
    """
    logger.info(f"Loading transactions data from: {filepath}")
    logger.info("Note: this file is large — may take 20-60 seconds...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"transactions_train.csv not found at: {filepath}\n"
            f"Make sure the H&M dataset is placed in data/raw/"
        )

    # Specify dtypes upfront to avoid incorrect inference and reduce memory
    # article_id and customer_id must be read as strings to preserve leading zeros
    dtype_map = {
        'customer_id': str,
        'article_id':  str,
        'sales_channel_id': 'int8',  # Only 1 or 2 — int8 saves memory
    }

    df = pd.read_csv(filepath, dtype=dtype_map)

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory usage: {mem_mb:.1f} MB")

    return df


# =============================================================================
# SECTION 2 — DUPLICATE REMOVAL
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate transaction records.

    What counts as a duplicate transaction?
        Same customer buying the same article on the same date = duplicate event.
        This is defined by: (customer_id, article_id, t_dat).

    Note on legitimate same-day same-item purchases:
        A customer could theoretically buy the same item twice on the same day.
        Without a unique transaction_id to distinguish these, we treat them as
        duplicates (data entry errors are more likely than genuine re-purchases).
        This is a safe assumption for training data; revisit if business logic differs.

    Args:
        df: Raw transactions DataFrame

    Returns:
        DataFrame with duplicate transaction events removed
    """
    before = len(df)

    # Remove exact full-row duplicates first (definitely errors)
    df = df.drop_duplicates()
    full_dup_removed = before - len(df)

    # Remove key-based duplicates
    before_key = len(df)
    key_cols = ['customer_id', 'article_id', 't_dat']
    # Only use key columns that actually exist in this dataset
    existing_key_cols = [c for c in key_cols if c in df.columns]
    df = df.drop_duplicates(subset=existing_key_cols, keep='first')
    key_dup_removed = before_key - len(df)

    total_removed = before - len(df)
    if total_removed > 0:
        logger.info(f"Removed {full_dup_removed:,} full-row duplicates")
        logger.info(f"Removed {key_dup_removed:,} key-based duplicates")
        logger.info(f"Total removed: {total_removed:,} rows ({total_removed/before*100:.3f}%)")
    else:
        logger.info("No duplicates found in transactions data")

    return df


# =============================================================================
# SECTION 3 — HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in transaction records.

    Transactions are interaction events — partial records are unusable.
    The primary IDs (customer_id, article_id, t_dat) are non-negotiable.

    Strategy:
        - customer_id, article_id, t_dat : Drop row if any is null
          (without these three, a transaction has no meaning)
        - price        : Fill with median price (missing price ≠ invalid transaction)
        - sales_channel_id : Fill with mode (most common channel)

    Why drop instead of fill for IDs?
        In collaborative filtering, every row in the interaction matrix must be
        attributable to a real user-item pair. An unknown customer or article
        cannot be used for training or evaluation.

    Args:
        df: DataFrame after duplicate removal

    Returns:
        DataFrame with nulls handled
    """
    logger.info("Handling missing values in transactions data...")

    # Drop rows with null in any critical column
    critical_cols = ['customer_id', 'article_id', 't_dat']
    existing_critical = [c for c in critical_cols if c in df.columns]

    before = len(df)
    df = df.dropna(subset=existing_critical)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} rows with null critical columns")
    else:
        logger.info("  No nulls found in critical columns (customer_id, article_id, t_dat)")

    # Fill price nulls with median price
    if 'price' in df.columns:
        null_price = df['price'].isnull().sum()
        if null_price > 0:
            median_price = df['price'].median()
            df['price'] = df['price'].fillna(median_price)
            logger.info(f"  'price': filled {null_price:,} nulls → median (${median_price:.4f})")

    # Fill sales_channel_id nulls with mode
    if 'sales_channel_id' in df.columns:
        null_channel = df['sales_channel_id'].isnull().sum()
        if null_channel > 0:
            mode_channel = df['sales_channel_id'].mode()[0]
            df['sales_channel_id'] = df['sales_channel_id'].fillna(mode_channel)
            logger.info(f"  'sales_channel_id': filled {null_channel:,} nulls → mode ({mode_channel})")

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Missing value handling complete. Remaining nulls: {remaining_nulls}")

    return df


# =============================================================================
# SECTION 4 — DATE CONVERSION
# =============================================================================

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the t_dat column from string to datetime.

    Why this is critical for recommendation systems:
        1. TEMPORAL SPLITTING: We split train/test by time (e.g., last week = test).
           This requires a real datetime type for comparison.
        2. RECENCY FEATURES: Days since last purchase is one of the strongest
           features in recommendation models (recency signal).
        3. SEASONALITY: Month and week features capture fashion trend cycles.
        4. CHRONOLOGICAL SORT: All time-series operations require proper datetime.

    Args:
        df: DataFrame after null handling

    Returns:
        DataFrame with t_dat as datetime64 type
    """
    logger.info("Converting date column to datetime...")

    if 't_dat' not in df.columns:
        logger.warning("'t_dat' column not found — skipping date conversion")
        return df

    before_dtype = df['t_dat'].dtype
    df['t_dat'] = pd.to_datetime(df['t_dat'])

    date_min = df['t_dat'].min()
    date_max = df['t_dat'].max()
    date_range_days = (date_max - date_min).days

    logger.info(f"  Converted: {before_dtype} → {df['t_dat'].dtype}")
    logger.info(f"  Date range: {date_min.date()} → {date_max.date()} ({date_range_days} days)")

    return df


# =============================================================================
# SECTION 5 — FIX DATA TYPES & NORMALIZE IDs
# =============================================================================

def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns have correct, consistent data types.

    ID normalization is critical:
        customer_id and article_id must have the same format here as in
        customers_cleaned.csv and articles_cleaned.csv — otherwise joins will fail silently.

    Args:
        df: DataFrame after date conversion

    Returns:
        DataFrame with normalized types
    """
    logger.info("Fixing data types in transactions data...")

    # Normalize string IDs — strip whitespace to prevent invisible join failures
    for id_col in ['customer_id', 'article_id']:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str).str.strip()

    # price should be float
    if 'price' in df.columns:
        df['price'] = df['price'].astype(float)

    # sales_channel_id is a small integer (1 or 2) — int8 is sufficient
    if 'sales_channel_id' in df.columns:
        df['sales_channel_id'] = df['sales_channel_id'].astype('int8')

    logger.info("Data types fixed successfully")
    return df


# =============================================================================
# SECTION 6 — CHRONOLOGICAL SORT
# =============================================================================

def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort transactions by date from oldest to newest.

    Why chronological order matters:
        - Time-based train/test split requires sorted data
          (e.g., "use all transactions before 2020-09-01 for training")
        - Recency features (last N purchases) require correct ordering
        - Prevents data leakage: future transactions must never appear in training

    Args:
        df: DataFrame after type normalization

    Returns:
        DataFrame sorted chronologically by t_dat, index reset
    """
    logger.info("Sorting transactions chronologically...")

    df = df.sort_values('t_dat', ascending=True).reset_index(drop=True)

    logger.info(f"  First transaction : {df['t_dat'].iloc[0]}")
    logger.info(f"  Last transaction  : {df['t_dat'].iloc[-1]}")

    return df


# =============================================================================
# SECTION 7 — SAVE OUTPUT
# =============================================================================

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned transactions DataFrame to the processed/ directory.

    Args:
        df: Fully cleaned transactions DataFrame
        output_path: Destination file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Saving cleaned transactions... (large file, may take a moment)")
    df.to_csv(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"Saved cleaned transactions → {output_path}")
    logger.info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Size: {file_size_mb:.1f} MB")


# =============================================================================
# SECTION 8 — DATA QUALITY REPORT
# =============================================================================

def generate_quality_report(df: pd.DataFrame) -> None:
    """
    Print a final quality summary after all cleaning steps.

    This serves as a final sanity check before the data moves to
    the feature engineering phase. Any unexpected values here
    indicate a bug in the preprocessing logic.

    Args:
        df: Fully cleaned transactions DataFrame
    """
    logger.info("=" * 60)
    logger.info("  TRANSACTIONS QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"  Total transactions    : {len(df):,}")
    logger.info(f"  Unique customers      : {df['customer_id'].nunique():,}")
    logger.info(f"  Unique articles       : {df['article_id'].nunique():,}")
    logger.info(f"  Date range            : {df['t_dat'].min().date()} → {df['t_dat'].max().date()}")
    logger.info(f"  Remaining null values : {df.isnull().sum().sum()}")

    if 'price' in df.columns:
        logger.info(f"  Price — min: ${df['price'].min():.4f}  |  max: ${df['price'].max():.4f}  |  median: ${df['price'].median():.4f}")

    if 'sales_channel_id' in df.columns:
        channel_counts = df['sales_channel_id'].value_counts().to_dict()
        logger.info(f"  Sales channels        : {channel_counts}")

    logger.info("=" * 60)


# =============================================================================
# SECTION 9 — MAIN PIPELINE FUNCTION
# =============================================================================

def preprocess_transactions(
    input_path: str  = RAW_DATA_PATH,
    output_path: str = OUTPUT_PATH
) -> pd.DataFrame:
    """
    Main entry point — runs the full transactions preprocessing pipeline.

    Pipeline order:
        1. Load raw data
        2. Remove duplicates
        3. Handle missing values
        4. Convert dates to datetime
        5. Fix data types
        6. Sort chronologically
        7. Generate quality report
        8. Save cleaned output

    Args:
        input_path:  Path to raw transactions_train.csv
        output_path: Path where cleaned CSV will be saved

    Returns:
        Cleaned transactions DataFrame (also saved to disk)
    """
    logger.info("=" * 60)
    logger.info("  TRANSACTIONS PREPROCESSING PIPELINE — START")
    logger.info("=" * 60)

    start_time = datetime.now()

    df = load_raw_transactions(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_dates(df)
    df = fix_datatypes(df)
    df = sort_chronologically(df)
    generate_quality_report(df)
    save_cleaned_data(df, output_path)

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info(f"  TRANSACTIONS PREPROCESSING COMPLETE in {elapsed:.2f}s")
    logger.info("=" * 60)

    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    cleaned_df = preprocess_transactions()

    print("\n--- Cleaned Transactions Preview (first 5 rows) ---")
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")
    print(f"Null count : {cleaned_df.isnull().sum().sum()}")
