"""
query_tower.py
══════════════════════════════════════════════════════════════════════
Query Tower (User Encoder) for the Two-Tower Retrieval Model.

WHAT IS THE QUERY TOWER?
    The Query Tower is the "user side" of the Two-Tower architecture.
    It converts a user ID (and optionally more user features) into a
    fixed-size embedding vector that represents the user's preferences.

    At inference time, this embedding is compared against all item
    embeddings to find the most relevant items for that user.

ARCHITECTURE:
    user_id (string)
        → StringLookup   (string → integer index)
        → Embedding      (integer → dense vector)
        → [Dense layers] (optional: deeper representations)
        → user_embedding (final output vector)

USAGE:
    from src.models.query_tower import QueryTower

    tower = QueryTower(user_vocabulary=vocab, embedding_dim=32)
    embedding = tower(tf.constant(["user_001"]))
    # embedding.shape → (1, 32)

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────
# QUERY TOWER CLASS
# ─────────────────────────────────────────────────────────────────────

class QueryTower(tf.keras.Model):
    """
    Query Tower (User Encoder) for Two-Tower retrieval.

    Encodes user IDs into dense embedding vectors. The embeddings
    are trained so that users with similar preferences end up close
    together in embedding space (high cosine / dot-product similarity).

    Args:
        user_vocabulary (tf.keras.layers.StringLookup):
            Pre-built vocabulary layer that maps user_id strings → ints.
        embedding_dim (int):
            Dimensionality of the output embedding vector.
            Typical values: 16, 32, 64, 128.
            Larger = more expressive but slower. Start with 32.
        use_dense_layers (bool):
            If True, adds Dense layers after the embedding for richer
            representations. Set to False for the simplest v1 model.
        dense_units (list of int):
            Hidden units for optional Dense layers, e.g. [64, 32].
            Only used when use_dense_layers=True.
    """

    def __init__(
        self,
        user_vocabulary: tf.keras.layers.StringLookup,
        embedding_dim: int = 32,
        use_dense_layers: bool = False,
        dense_units: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim     = embedding_dim
        self.use_dense_layers  = use_dense_layers

        # ── Layer 1: String → Integer ──────────────────────────────
        # StringLookup maps user_id strings to integer indices.
        # The vocabulary was built from all unique user IDs in the dataset.
        self.user_lookup = user_vocabulary

        # ── Layer 2: Integer → Dense Vector ───────────────────────
        # Embedding layer: a learnable lookup table.
        # Shape of the weight matrix: (num_users, embedding_dim)
        # Each row is the embedding for one user (learned during training).
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=user_vocabulary.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_initializer="glorot_uniform",
            name="user_embedding_weights",
        )

        # ── Layer 3 (optional): Dense Transformation ────────────────
        # For v1 we skip dense layers to keep the model simple.
        # In future iterations, add Dense layers to let the model
        # learn non-linear transformations of the embedding space.
        self.dense_layers = []
        if use_dense_layers and dense_units:
            for i, units in enumerate(dense_units):
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        units=units,
                        activation="relu",
                        name=f"user_dense_{i}",
                    )
                )
            # Final projection to embedding_dim (ensures consistent output size)
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units=embedding_dim,
                    activation=None,    # No activation on final layer
                    name="user_projection",
                )
            )

    def call(self, user_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass: user_id strings → user embedding vectors.

        Args:
            user_ids (tf.Tensor):
                String tensor of user IDs, shape: (batch_size,)
                Example: tf.constant(["user_001", "user_042"])
            training (bool):
                Whether the model is in training mode.
                Affects dropout, BatchNorm, etc. (if added later).

        Returns:
            tf.Tensor: User embeddings, shape: (batch_size, embedding_dim)
        """
        # Step 1: Convert string IDs to integer indices
        # "user_001" → 0,  "user_042" → 41,  etc.
        user_indices = self.user_lookup(user_ids)         # (batch_size,)

        # Step 2: Look up embedding for each integer index
        # The embedding layer returns the learned vector for each index
        x = self.user_embedding(user_indices)             # (batch_size, embedding_dim)

        # Step 3 (optional): Pass through Dense layers
        if self.use_dense_layers:
            for dense_layer in self.dense_layers:
                x = dense_layer(x)

        return x                                          # (batch_size, embedding_dim)

    def get_config(self) -> dict:
        """
        Serialize model configuration for saving/loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "embedding_dim"    : self.embedding_dim,
            "use_dense_layers" : self.use_dense_layers,
        })
        return config


# ─────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# Convenience wrapper that builds the tower from a vocabulary list
# ─────────────────────────────────────────────────────────────────────

def build_query_tower(
    user_ids: list,
    embedding_dim: int = 32,
    use_dense_layers: bool = False,
    dense_units: list = None,
) -> QueryTower:
    """
    Build a QueryTower from a list of unique user IDs.

    Internally creates the StringLookup vocabulary layer, then
    instantiates and returns the QueryTower.

    Args:
        user_ids (list of str):
            All unique user IDs in the dataset.
        embedding_dim (int):
            Dimensionality of the output embedding vector.
        use_dense_layers (bool):
            Whether to add Dense layers after the embedding.
        dense_units (list of int):
            Hidden units for Dense layers (if use_dense_layers=True).

    Returns:
        QueryTower: A ready-to-use query tower model.

    Example:
        tower = build_query_tower(
            user_ids=user_df["user_id"].unique().tolist(),
            embedding_dim=32
        )
    """
    # Build vocabulary: maps string IDs → integer indices
    vocabulary = tf.keras.layers.StringLookup(
        vocabulary=user_ids,
        mask_token=None,        # No mask token (we don't need padding for IDs)
        name="user_vocabulary",
    )

    # Instantiate and return the tower
    tower = QueryTower(
        user_vocabulary=vocabulary,
        embedding_dim=embedding_dim,
        use_dense_layers=use_dense_layers,
        dense_units=dense_units or [],
        name="query_tower",
    )

    return tower
