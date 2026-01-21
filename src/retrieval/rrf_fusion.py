"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

RRF is a simple but effective method for fusing ranked lists from different
retrieval methods (e.g., BM25 and dense vector search).

Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
"Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
"""

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger()


def reciprocal_rank_fusion(
    results_lists: list[list[tuple[Document, float]]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[Document, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF Score for a document = sum(1 / (k + rank_i)) for each list where
    the document appears.

    Args:
        results_lists: List of result lists, where each result list contains
            (Document, score) tuples sorted by descending score.
        k: RRF constant. Higher values reduce the impact of rank differences.
            Default is 60 (standard value from the original paper).
        top_n: Maximum number of documents to return.

    Returns:
        List of (Document, rrf_score) tuples sorted by descending RRF score.
    """
    if not results_lists:
        return []

    # Use content hash as document ID for deduplication
    def doc_id(doc: Document) -> int:
        return hash(doc.page_content)

    # Track RRF scores and document references
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for results in results_lists:
        for rank, (doc, _score) in enumerate(results):
            did = doc_id(doc)

            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed, so rank 0 (best) gets score 1/(k+0) = 1/k
            rrf_score = 1.0 / (k + rank)

            if did in rrf_scores:
                rrf_scores[did] += rrf_score
            else:
                rrf_scores[did] = rrf_score
                doc_map[did] = doc

    # Sort by RRF score (descending) and return top_n
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = [(doc_map[did], score) for did, score in sorted_results[:top_n]]

    logger.debug(f"RRF fused {sum(len(r) for r in results_lists)} candidates into {len(fused_results)} results")

    return fused_results


def weighted_rrf(
    results_lists: list[list[tuple[Document, float]]],
    weights: list[float] | None = None,
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[Document, float]]:
    """
    Weighted Reciprocal Rank Fusion.

    Like standard RRF but allows different weights for each result list.

    Args:
        results_lists: List of result lists, where each result list contains
            (Document, score) tuples sorted by descending score.
        weights: Optional weights for each result list. If None, all lists
            are weighted equally.
        k: RRF constant.
        top_n: Maximum number of documents to return.

    Returns:
        List of (Document, weighted_rrf_score) tuples sorted by descending score.
    """
    if not results_lists:
        return []

    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(results_lists)

    if len(weights) != len(results_lists):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of result lists ({len(results_lists)})")

    def doc_id(doc: Document) -> int:
        return hash(doc.page_content)

    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for weight, results in zip(weights, results_lists, strict=False):
        for rank, (doc, _score) in enumerate(results):
            did = doc_id(doc)

            # Weighted RRF formula
            rrf_score = weight * (1.0 / (k + rank))

            if did in rrf_scores:
                rrf_scores[did] += rrf_score
            else:
                rrf_scores[did] = rrf_score
                doc_map[did] = doc

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[did], score) for did, score in sorted_results[:top_n]]
