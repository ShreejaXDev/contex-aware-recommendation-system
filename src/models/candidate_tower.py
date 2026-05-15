"""
candidate_tower.py
══════════════════════════════════════════════════════════════════════
Candidate Tower (Item Encoder) for the Two-Tower Retrieval Model.

WHAT IS THE CANDIDATE TOWER?
    The Candidate Tower is the "item side" of the Two-Tower architecture.
    It converts an item ID into a fixed-size embedding vector that
    represents the item's characteristics.

    These item embeddings are pre-computed offline and stored in an
    index (e.g., FAISS, ScaNN, or BruteForce for small catalogs).
    At inference time, we compare a user embedding against all stored
    item embeddings to find the best matches.

ARCHITECTURE:
    item_id (string)
        → StringLookup   (string → integer index)
        → Embedding      (integer → dense vector)
        → [Dense layers] (optional: deeper representations)
        → item_embedding (final output vector)

WHY THE SAME EMBEDDING DIM AS USER TOWER?
    Both towers must output vectors of the SAME dimension so we can
    compute dot-product similarity between them.
    Dot product: user_vec · item_vec = similarity score

USAGE:
    from src.models.candidate_tower import CandidateTower

    tower = CandidateTower(item_vocabulary=vocab, embedding_dim=32)
    embedding = tower(tf.constant(["PROD_001"]))
    # embedding.shape → (1, 32)

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────
# CANDIDATE TOWER CLASS
# ─────────────────────────────────────────────────────────────────────

class CandidateTower(tf.keras.Model):
    """
    Candidate Tower (Item Encoder) for Two-Tower retrieval.

    Encodes item IDs into dense embedding vectors. The embeddings are
    trained so items that similar users interact with end up close
    together in embedding space.

    Args:
        item_vocabulary (tf.keras.layers.StringLookup):
            Pre-built vocabulary layer that maps item_id strings → ints.
        embedding_dim (int):
            Dimensionality of the output embedding vector.
            MUST match the QueryTower's embedding_dim for dot-product
            similarity to work correctly.
        use_dense_layers (bool):
            If True, adds Dense layers after the embedding.
        dense_units (list of int):
            Hidden units for optional Dense layers, e.g. [64, 32].
    """

    def __init__(
        self,
        item_vocabulary: tf.keras.layers.StringLookup,
        embedding_dim: int = 32,
        use_dense_layers: bool = False,
        dense_units: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim    = embedding_dim
        self.use_dense_layers = use_dense_layers

        # ── Layer 1: String → Integer ──────────────────────────────
        # Same pattern as QueryTower: string ID → integer index
        self.item_lookup = item_vocabulary

        # ── Layer 2: Integer → Dense Vector ───────────────────────
        # Separate embedding matrix from the user embedding.
        # Shape: (num_items, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=item_vocabulary.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_initializer="glorot_uniform",
            name="item_embedding_weights",
        )

        # ── Layer 3 (optional): Dense Transformation ────────────────
        # For v1 we keep it simple. Future versions can incorporate:
        #   - Item category embeddings
        #   - Item price features (normalized)
        #   - Item text features (from product descriptions)
        self.dense_layers = []
        if use_dense_layers and dense_units:
            for i, units in enumerate(dense_units):
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        units=units,
                        activation="relu",
                        name=f"item_dense_{i}",
                    )
                )
            # Project back to embedding_dim for consistent output size
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units=embedding_dim,
                    activation=None,
                    name="item_projection",
                )
            )

    def call(self, item_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass: item_id strings → item embedding vectors.

        Args:
            item_ids (tf.Tensor):
                String tensor of item IDs, shape: (batch_size,)
                Example: tf.constant(["PROD_001", "PROD_099"])
            training (bool):
                Whether the model is in training mode.

        Returns:
            tf.Tensor: Item embeddings, shape: (batch_size, embedding_dim)
        """
        # Step 1: String ID → integer index
        item_indices = self.item_lookup(item_ids)          # (batch_size,)

        # Step 2: Integer index → embedding vector
        x = self.item_embedding(item_indices)              # (batch_size, embedding_dim)

        # Step 3 (optional): Dense layers
        if self.use_dense_layers:
            for dense_layer in self.dense_layers:
                x = dense_layer(x)

        return x                                           # (batch_size, embedding_dim)

    def get_config(self) -> dict:
        """Serialize config for saving/loading."""
        config = super().get_config()
        config.update({
            "embedding_dim"   : self.embedding_dim,
            "use_dense_layers": self.use_dense_layers,
        })
        return config


# ─────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────

def build_candidate_tower(
    item_ids: list,
    embedding_dim: int = 32,
    use_dense_layers: bool = False,
    dense_units: list = None,
) -> CandidateTower:
    """
    Build a CandidateTower from a list of unique item IDs.

    Args:
        item_ids (list of str):
            All unique item IDs in the product catalog.
        embedding_dim (int):
            Dimensionality of the output embedding vector.
            Must match the QueryTower's embedding_dim.
        use_dense_layers (bool):
            Whether to add Dense layers after the embedding.
        dense_units (list of int):
            Hidden units for Dense layers (if use_dense_layers=True).

    Returns:
        CandidateTower: A ready-to-use candidate tower model.

    Example:
        tower = build_candidate_tower(
            item_ids=item_df["item_id"].unique().tolist(),
            embedding_dim=32
        )
    """
    vocabulary = tf.keras.layers.StringLookup(
        vocabulary=item_ids,
        mask_token=None,
        name="item_vocabulary",
    )

    tower = CandidateTower(
        item_vocabulary=vocabulary,
        embedding_dim=embedding_dim,
        use_dense_layers=use_dense_layers,
        dense_units=dense_units or [],
        name="candidate_tower",
    )

    return tower
