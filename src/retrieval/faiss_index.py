"""
faiss_index.py
====================================================================
FAISS-based ANN retrieval module for the Two-Tower recommendation system.

This module provides a small, production-style abstraction that:
- Builds a FAISS index over item embeddings (ANN search)
- Supports top-K retrieval for a single user or a batch of users
- Includes a BruteForce baseline for optional performance comparison
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


def _require_faiss():
    """Ensure FAISS is installed before building a FAISS index."""
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError(
            "FAISS is not installed. Install it with: pip install faiss-cpu"
        ) from exc

    return faiss


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity via inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _build_item_embeddings(
    item_ids: List[str],
    candidate_tower,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """Compute item embeddings using the candidate tower."""
    if not item_ids:
        raise ValueError("Item ID list is empty; cannot build retrieval index.")

    ids_tensor = tf.constant(item_ids)
    dataset = tf.data.Dataset.from_tensor_slices(ids_tensor).batch(batch_size)

    embeddings: List[np.ndarray] = []
    for batch in dataset:
        batch_emb = candidate_tower(batch)
        embeddings.append(batch_emb.numpy())

    all_embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    if normalize:
        all_embeddings = _l2_normalize(all_embeddings)

    return all_embeddings


def _build_bruteforce_index(
    query_tower,
    candidate_tower,
    item_ids: List[str],
    top_k: int,
    batch_size: int,
) -> tfrs.layers.factorized_top_k.BruteForce:
    """Build a BruteForce retrieval index for baseline comparisons."""
    ids_tensor = tf.constant(item_ids)
    item_ds = tf.data.Dataset.from_tensor_slices(ids_tensor)
    item_ds = tf.data.Dataset.zip((item_ds, item_ds.map(candidate_tower))).batch(
        batch_size
    )

    index = tfrs.layers.factorized_top_k.BruteForce(
        query_model=query_tower,
        k=top_k,
    )
    index.index_from_dataset(item_ds)
    return index


class FaissRetrievalIndex:
    """
    FAISS ANN retrieval index wrapper.

    Args:
        query_tower: Trained query tower.
        candidate_tower: Trained candidate tower.
        item_ids: List of all item ID strings.
        top_k: Default number of recommendations to return.
        batch_size: Batch size for embedding computation.
        use_ivf: Use IVF index for larger catalogs (optional).
        ivf_nlist: Number of IVF clusters.
        ivf_nprobe: Number of clusters to search.
        normalize: L2-normalize embeddings (cosine similarity via inner product).
    """

    def __init__(
        self,
        query_tower,
        candidate_tower,
        item_ids: List[str],
        top_k: int = 10,
        batch_size: int = 256,
        use_ivf: bool = False,
        ivf_nlist: int = 100,
        ivf_nprobe: int = 10,
        normalize: bool = False,
    ):
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.item_ids = item_ids
        self.top_k = top_k
        self.batch_size = batch_size
        self.use_ivf = use_ivf
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.normalize = normalize

        self._faiss = _require_faiss()
        self._item_embeddings = _build_item_embeddings(
            item_ids=self.item_ids,
            candidate_tower=self.candidate_tower,
            batch_size=self.batch_size,
            normalize=self.normalize,
        )

        self._index = self._build_index()

    def _build_index(self):
        """Build the FAISS index and add item embeddings."""
        dim = self._item_embeddings.shape[1]

        if self.use_ivf:
            quantizer = self._faiss.IndexFlatIP(dim)
            index = self._faiss.IndexIVFFlat(
                quantizer,
                dim,
                self.ivf_nlist,
                self._faiss.METRIC_INNER_PRODUCT,
            )
            index.train(self._item_embeddings)
            index.nprobe = self.ivf_nprobe
        else:
            index = self._faiss.IndexFlatIP(dim)

        index.add(self._item_embeddings)
        return index

    def _embed_user(self, user_id: str) -> np.ndarray:
        """Compute the user embedding used for ANN search."""
        user_vec = self.query_tower(tf.constant([user_id])).numpy().astype("float32")
        if self.normalize:
            user_vec = _l2_normalize(user_vec)
        return user_vec

    def retrieve(self, user_id: str, top_k: Optional[int] = None) -> Tuple[List[float], List[str]]:
        """Return (scores, item_ids) for a single user."""
        k = top_k or self.top_k
        user_vec = self._embed_user(user_id)
        scores, indices = self._index.search(user_vec, k)

        scores_list = scores[0].tolist() if scores.size else []
        indices_list = indices[0].tolist() if indices.size else []

        item_ids = []
        filtered_scores = []
        for score, idx in zip(scores_list, indices_list):
            if idx < 0:
                continue
            item_ids.append(self.item_ids[idx])
            filtered_scores.append(score)

        return filtered_scores, item_ids

    def recommend(self, user_id: str, top_k: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame with columns [user_id, item_id, score, rank]."""
        scores, item_ids = self.retrieve(user_id, top_k=top_k)

        return pd.DataFrame(
            {
                "user_id": user_id,
                "item_id": item_ids,
                "score": scores,
                "rank": list(range(1, len(item_ids) + 1)),
            }
        )

    def recommend_batch(self, user_ids: Iterable[str], top_k: Optional[int] = None) -> pd.DataFrame:
        """Generate recommendations for multiple users and stack results."""
        k = top_k or self.top_k
        all_recs = []

        for user_id in user_ids:
            scores, item_ids = self.retrieve(user_id, top_k=k)
            all_recs.append(
                pd.DataFrame(
                    {
                        "user_id": user_id,
                        "item_id": item_ids,
                        "score": scores,
                        "rank": list(range(1, len(item_ids) + 1)),
                    }
                )
            )

        if not all_recs:
            return pd.DataFrame(columns=["user_id", "item_id", "score", "rank"])

        return pd.concat(all_recs, ignore_index=True)


def compare_retrieval_performance(
    query_tower,
    candidate_tower,
    item_ids: List[str],
    user_ids: List[str],
    top_k: int = 10,
    batch_size: int = 256,
    use_ivf: bool = False,
) -> Optional[str]:
    """
    Compare retrieval latency between BruteForce and FAISS.

    Returns a formatted string.
    """
    if not user_ids:
        return "No users available for performance comparison."

    sample_user = user_ids[0]

    brute_index = _build_bruteforce_index(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        item_ids=item_ids,
        top_k=top_k,
        batch_size=batch_size,
    )

    faiss_index = FaissRetrievalIndex(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        item_ids=item_ids,
        top_k=top_k,
        batch_size=batch_size,
        use_ivf=use_ivf,
    )

    start = time.perf_counter()
    _ = brute_index(tf.constant([sample_user]), k=top_k)
    brute_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = faiss_index.retrieve(sample_user, top_k=top_k)
    faiss_time = time.perf_counter() - start

    speedup = (brute_time / faiss_time) if faiss_time > 0 else float("inf")

    return (
        "=" * 50
        + "\nRETRIEVAL PERFORMANCE COMPARISON\n"
        + "=" * 50
        + f"\nBruteForce Retrieval : {brute_time:.4f} sec"
        + f"\nFAISS Retrieval      : {faiss_time:.4f} sec"
        + f"\nSpeed Improvement    : {speedup:.1f}x\n"
        + "=" * 50
    )
