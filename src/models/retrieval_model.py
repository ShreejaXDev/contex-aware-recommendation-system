"""
retrieval_model.py
══════════════════════════════════════════════════════════════════════
Full Two-Tower Retrieval Model.

WHAT IS THIS FILE?
    This is the "glue" that combines the QueryTower and CandidateTower
    into a single trainable model using TensorFlow Recommenders (TFRS).

HOW DOES THE RETRIEVAL TASK WORK?
    TFRS Retrieval uses a technique called "in-batch negative sampling":

    During training, a batch of interactions looks like:
        (User A, Item 3)  ← real interaction (positive)
        (User B, Item 7)  ← real interaction (positive)
        (User C, Item 1)  ← real interaction (positive)

    The model also considers cross-pairs as negatives:
        (User A, Item 7)  ← probably not interacted (negative)
        (User A, Item 1)  ← probably not interacted (negative)
        (User B, Item 3)  ← probably not interacted (negative)
        ... etc.

    Goal: maximize score(user, interacted_item) vs all negatives
    Loss: Categorical Cross-Entropy (softmax over all items in batch)

    EVALUATION METRIC: FactorizedTopK
        Measures: "Does the true item appear in our Top-K predictions?"
        Reported as: top_1, top_5, top_10, top_50, top_100 accuracy

USAGE:
    from src.models.retrieval_model import TwoTowerRetrievalModel

    model = TwoTowerRetrievalModel(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        items_dataset=items_ds
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(train_ds, epochs=5)

PROJECT: Context-Aware Neural Recommendation Engine
PHASE  : Core Deep Learning — Two-Tower Retrieval
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs


# ─────────────────────────────────────────────────────────────────────
# TWO-TOWER RETRIEVAL MODEL CLASS
# ─────────────────────────────────────────────────────────────────────

class TwoTowerRetrievalModel(tfrs.models.Model):
    """
    Full Two-Tower Retrieval Model combining user and item encoders.

    Inherits from tfrs.models.Model which provides:
    - Automatic loss routing via compute_loss()
    - Built-in integration with Keras model.fit() / model.evaluate()
    - Support for TFRS tasks (Retrieval, Ranking)

    Args:
        query_tower (QueryTower):
            The user encoder that maps user_ids → user embeddings.
        candidate_tower (CandidateTower):
            The item encoder that maps item_ids → item embeddings.
        items_dataset (tf.data.Dataset):
            Dataset of ALL item IDs used to build the evaluation index.
            The FactorizedTopK metric scores the true item against ALL
            items in this dataset — so include your full catalog.
    """

    def __init__(
        self,
        query_tower,
        candidate_tower,
        items_dataset: tf.data.Dataset,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── Sub-models (the two towers) ───────────────────────────
        self.query_tower     = query_tower
        self.candidate_tower = candidate_tower

        # ── Retrieval Task ────────────────────────────────────────
        # tfrs.tasks.Retrieval handles:
        #   1. Computing in-batch negative sampling loss
        #   2. Evaluating FactorizedTopK (Top-1, 5, 10, 50, 100)
        #
        # FactorizedTopK needs all item embeddings pre-computed.
        # It maps the entire items_dataset through candidate_tower.
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items_dataset.map(candidate_tower)
            )
        )

    def compute_loss(
        self,
        features: dict,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Compute retrieval loss for a batch of user-item interactions.

        Called automatically by model.fit() and model.evaluate().

        Args:
            features (dict):
                Batch of interaction data with keys:
                    "user_id" → string tensor, shape (batch_size,)
                    "item_id" → string tensor, shape (batch_size,)
            training (bool):
                True during model.fit(), False during model.evaluate().
                We skip metric computation during training for speed.

        Returns:
            tf.Tensor: Scalar loss value. The optimizer minimizes this.
        """
        # ── Step 1: Encode users ──────────────────────────────────
        # Shape: (batch_size, embedding_dim)
        user_embeddings = self.query_tower(
            features["user_id"], training=training
        )

        # ── Step 2: Encode items ──────────────────────────────────
        # Shape: (batch_size, embedding_dim)
        item_embeddings = self.candidate_tower(
            features["item_id"], training=training
        )

        # ── Step 3: Compute retrieval loss ────────────────────────
        # This internally computes:
        #   scores = user_embeddings @ item_embeddings.T  (dot product matrix)
        #   loss   = CrossEntropy(scores, diagonal_labels)
        #   where diagonal = [0,1,2,...,batch_size-1] (true pairs)
        loss = self.retrieval_task(
            query_embeddings=user_embeddings,
            candidate_embeddings=item_embeddings,
            compute_metrics=not training,  # Only compute metrics on eval
        )

        return loss

    def get_user_embedding(self, user_id: str) -> tf.Tensor:
        """
        Get the embedding vector for a single user.

        Useful for:
        - Debugging: inspect what the model learned about a user
        - Serving: pass embedding to nearest-neighbor index

        Args:
            user_id (str): Single user ID string.

        Returns:
            tf.Tensor: User embedding, shape (1, embedding_dim)
        """
        return self.query_tower(tf.constant([user_id]))

    def get_item_embedding(self, item_id: str) -> tf.Tensor:
        """
        Get the embedding vector for a single item.

        Args:
            item_id (str): Single item ID string.

        Returns:
            tf.Tensor: Item embedding, shape (1, embedding_dim)
        """
        return self.candidate_tower(tf.constant([item_id]))


# ─────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────

def build_retrieval_model(
    query_tower,
    candidate_tower,
    items_dataset: tf.data.Dataset,
) -> TwoTowerRetrievalModel:
    """
    Instantiate and return a TwoTowerRetrievalModel.

    Args:
        query_tower     : QueryTower instance
        candidate_tower : CandidateTower instance
        items_dataset   : tf.data.Dataset of all item IDs (batched)

    Returns:
        TwoTowerRetrievalModel: Assembled retrieval model (not yet compiled).

    Example:
        model = build_retrieval_model(query_tower, candidate_tower, items_ds)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.1))
    """
    return TwoTowerRetrievalModel(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        items_dataset=items_dataset,
        name="two_tower_retrieval_model",
    )


# Backward-compatible alias used by the API entrypoint.
RetrievalModel = TwoTowerRetrievalModel
