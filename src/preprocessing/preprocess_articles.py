"""
preprocess_articles.py
======================
Production preprocessing script for the H&M articles dataset.

Purpose:
    - Load raw articles.csv
    - Clean and standardize all columns (product catalog data)
    - Handle missing values in categorical and text fields
    - Remove duplicate articles
    - Fix data types
    - Save cleaned output to data/processed/articles_cleaned.csv

Why articles preprocessing matters for recommendations:
    Articles are the ITEMS being recommended. Clean, consistent item metadata
    enables content-based filtering (recommending similar items) and ensures
    correct joins with transaction data. Dirty article data = wrong items shown.

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
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH  = os.path.join(BASE_DIR, 'data', 'raw',       'articles.csv')
OUTPUT_PATH    = os.path.join(BASE_DIR, 'data', 'processed', 'articles_cleaned.csv')


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_raw_articles(filepath: str) -> pd.DataFrame:
    """
    Load raw articles CSV from disk.

    Args:
        filepath: Absolute path to articles.csv

    Returns:
        Raw DataFrame (no modifications)

    Note:
        article_id can come in as an integer or string depending on pandas
        inference. We handle normalization in fix_datatypes().
    """
    logger.info(f"Loading articles data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"articles.csv not found at: {filepath}\n"
            f"Make sure the H&M dataset is placed in data/raw/"
        )

    # dtype={'article_id': str} prevents pandas from converting IDs to scientific notation
    df = pd.read_csv(filepath, dtype={'article_id': str})

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


# =============================================================================
# SECTION 2 — DUPLICATE REMOVAL
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate article records.

    Strategy:
        Each article_id should appear exactly once in the catalog.
        If duplicated, keep the first occurrence (data entry error assumed).

    Why duplicates are harmful for content-based filtering:
        If the same article appears twice, it gets double the weight in
        item-item similarity matrices, skewing recommendations toward it.

    Args:
        df: Raw articles DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    before = len(df)

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['article_id'], keep='first')

    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate article records")
    else:
        logger.info("No duplicates found in articles data")

    return df


# =============================================================================
# SECTION 3 — HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values across all article columns.

    The articles dataset contains many categorical descriptor columns
    (product type, department, colour, etc.) that describe each item.

    Strategy by column type:
        - article_id      : Drop rows (cannot identify item without this)
        - description     : Fill with 'No description' (avoid NLP errors downstream)
        - all other object: Fill with 'Unknown' (preserves record, valid category)
        - numeric columns : Fill with median (for any numerical product attributes)

    Why 'Unknown' instead of dropping?
        Dropping rows with missing category labels would remove valid articles
        from the catalog. The model can learn to treat 'Unknown' as its own category.

    Args:
        df: DataFrame after duplicate removal

    Returns:
        DataFrame with nulls handled
    """
    logger.info("Handling missing values in articles data...")

    # Drop rows with no article_id — cannot use these in recommendations
    before = len(df)
    df = df.dropna(subset=['article_id'])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} rows with null article_id")

    # ── Object (string/categorical) columns ───────────────────────────────────
    object_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove article_id from this list — handled separately
    object_cols = [c for c in object_cols if c != 'article_id']

    filled_summary = []
    for col in object_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            # Description columns get a more specific placeholder
            fill_value = 'No description' if 'desc' in col.lower() else 'Unknown'
            df[col] = df[col].fillna(fill_value)
            filled_summary.append(f"'{col}': {null_count:,} nulls → '{fill_value}'")

    if filled_summary:
        for msg in filled_summary:
            logger.info(f"  {msg}")
    else:
        logger.info("  No nulls found in categorical columns")

    # ── Numeric columns ───────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  '{col}': {null_count:,} nulls → median ({median_val})")

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Missing value handling complete. Remaining nulls: {remaining_nulls}")

    return df


# =============================================================================
# SECTION 4 — TEXT STANDARDIZATION
# =============================================================================

def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text formatting across all string columns.

    Why standardize text?
        'TROUSERS', 'Trousers', 'trousers ' are three different values to Python.
        This causes incorrect group counts and broken joins downstream.
        Consistent casing + stripped whitespace = reliable grouping.

    Steps:
        1. Strip leading/trailing whitespace
        2. Normalize to title case for categorical fields (looks clean too)
        3. Lowercase description fields (better for NLP later)

    Args:
        df: DataFrame after null handling

    Returns:
        DataFrame with standardized text
    """
    logger.info("Standardizing text columns...")

    object_cols = df.select_dtypes(include='object').columns.tolist()
    object_cols = [c for c in object_cols if c != 'article_id']  # don't alter IDs

    for col in object_cols:
        # Step 1: Strip whitespace
        df[col] = df[col].astype(str).str.strip()

        # Step 2: Apply appropriate casing
        if 'desc' in col.lower():
            # Description fields → lowercase (prep for NLP tokenization)
            df[col] = df[col].str.lower()
        else:
            # Category fields → title case (clean, consistent display)
            df[col] = df[col].str.title()

    logger.info(f"Standardized {len(object_cols)} text columns")
    return df


# =============================================================================
# SECTION 5 — FIX DATA TYPES
# =============================================================================

def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for all article columns.

    Args:
        df: DataFrame after text standardization

    Returns:
        DataFrame with corrected types
    """
    logger.info("Fixing data types in articles data...")

    # article_id MUST be a zero-padded string to match transaction references
    # H&M article IDs can have leading zeros that get dropped if stored as int
    if 'article_id' in df.columns:
        df['article_id'] = df['article_id'].astype(str).str.strip()

    # Integer code columns (e.g., product_type_no, graphical_appearance_no)
    # These are numeric category codes — should be integers, not floats
    int_code_cols = [c for c in df.columns if c.endswith('_no')]
    for col in int_code_cols:
        if col in df.columns and df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype('Int64')  # Int64 (capital I) supports NaN

    logger.info("Data types fixed successfully")
    return df


# =============================================================================
# SECTION 6 — SAVE OUTPUT
# =============================================================================

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned articles DataFrame to the processed/ directory.

    Args:
        df: Fully cleaned articles DataFrame
        output_path: Destination file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"Saved cleaned articles → {output_path}")
    logger.info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Size: {file_size_mb:.2f} MB")


# =============================================================================
# SECTION 7 — MAIN PIPELINE FUNCTION
# =============================================================================

def preprocess_articles(
    input_path: str  = RAW_DATA_PATH,
    output_path: str = OUTPUT_PATH
) -> pd.DataFrame:
    """
    Main entry point — runs the full articles preprocessing pipeline.

    Pipeline order:
        1. Load raw data
        2. Remove duplicates
        3. Handle missing values
        4. Standardize text columns
        5. Fix data types
        6. Save cleaned output

    Args:
        input_path:  Path to raw articles.csv
        output_path: Path where cleaned CSV will be saved

    Returns:
        Cleaned articles DataFrame (also saved to disk)
    """
    logger.info("=" * 60)
    logger.info("  ARTICLES PREPROCESSING PIPELINE — START")
    logger.info("=" * 60)

    start_time = datetime.now()

    df = load_raw_articles(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_text_columns(df)
    df = fix_datatypes(df)
    save_cleaned_data(df, output_path)

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info(f"  ARTICLES PREPROCESSING COMPLETE in {elapsed:.2f}s")
    logger.info("=" * 60)

    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    cleaned_df = preprocess_articles()

    print("\n--- Cleaned Articles Preview (first 5 rows) ---")
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")
    print(f"Null count : {cleaned_df.isnull().sum().sum()}")
