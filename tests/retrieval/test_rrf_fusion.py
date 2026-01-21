"""Tests for Reciprocal Rank Fusion."""

import pytest
from langchain_core.documents import Document

from src.retrieval.rrf_fusion import reciprocal_rank_fusion, weighted_rrf


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    @pytest.fixture
    def sample_docs(self):
        """Create sample documents."""
        return [
            Document(page_content="Document A about Python"),
            Document(page_content="Document B about JavaScript"),
            Document(page_content="Document C about Machine Learning"),
            Document(page_content="Document D about Web Development"),
            Document(page_content="Document E about Data Science"),
        ]

    def test_empty_input(self):
        """Should return empty list for empty input."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_list(self, sample_docs):
        """Should handle single result list."""
        results = [(sample_docs[0], 0.9), (sample_docs[1], 0.8), (sample_docs[2], 0.7)]

        fused = reciprocal_rank_fusion([results], k=60, top_n=3)

        assert len(fused) == 3
        # Order should be preserved for single list
        assert fused[0][0].page_content == sample_docs[0].page_content

    def test_two_lists_no_overlap(self, sample_docs):
        """Should combine results from two non-overlapping lists."""
        list1 = [(sample_docs[0], 0.9), (sample_docs[1], 0.8)]
        list2 = [(sample_docs[2], 0.85), (sample_docs[3], 0.75)]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=4)

        assert len(fused) == 4
        contents = [d.page_content for d, _ in fused]
        assert sample_docs[0].page_content in contents
        assert sample_docs[2].page_content in contents

    def test_two_lists_with_overlap(self, sample_docs):
        """Should boost documents appearing in multiple lists."""
        # Doc A appears first in both lists - should get highest score
        list1 = [(sample_docs[0], 0.9), (sample_docs[1], 0.8)]
        list2 = [(sample_docs[0], 0.95), (sample_docs[2], 0.7)]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)

        # Doc A should be first (appears in both lists at rank 0)
        assert fused[0][0].page_content == sample_docs[0].page_content
        # Its score should be higher than others
        assert fused[0][1] > fused[1][1]

    def test_top_n_limit(self, sample_docs):
        """Should respect top_n limit."""
        results = [(d, 0.5) for d in sample_docs]

        fused = reciprocal_rank_fusion([results], k=60, top_n=2)

        assert len(fused) == 2

    def test_rrf_scores_decrease_with_rank(self, sample_docs):
        """RRF scores should decrease for lower-ranked documents."""
        results = [(sample_docs[i], 1.0 - i * 0.1) for i in range(5)]

        fused = reciprocal_rank_fusion([results], k=60, top_n=5)

        # Scores should be in descending order
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_k_parameter_effect(self, sample_docs):
        """Different k values should produce different score distributions."""
        results = [(sample_docs[0], 0.9), (sample_docs[1], 0.8)]

        # Lower k = more weight to top ranks
        fused_low_k = reciprocal_rank_fusion([results], k=1, top_n=2)
        # Higher k = more even distribution
        fused_high_k = reciprocal_rank_fusion([results], k=100, top_n=2)

        # Score ratio should be different
        ratio_low = fused_low_k[0][1] / fused_low_k[1][1]
        ratio_high = fused_high_k[0][1] / fused_high_k[1][1]

        # Low k should have higher ratio (more differentiation)
        assert ratio_low > ratio_high

    def test_deduplication(self, sample_docs):
        """Should deduplicate documents based on content."""
        # Same document appearing multiple times in same list
        list1 = [(sample_docs[0], 0.9), (sample_docs[0], 0.8)]

        fused = reciprocal_rank_fusion([list1], k=60, top_n=5)

        # Should only appear once
        assert len(fused) == 1


class TestWeightedRRF:
    """Tests for the weighted_rrf function."""

    @pytest.fixture
    def sample_docs(self):
        """Create sample documents."""
        return [
            Document(page_content="Doc A"),
            Document(page_content="Doc B"),
            Document(page_content="Doc C"),
        ]

    def test_equal_weights(self, sample_docs):
        """Equal weights should give same results as standard RRF."""
        list1 = [(sample_docs[0], 0.9)]
        list2 = [(sample_docs[1], 0.8)]

        standard = reciprocal_rank_fusion([list1, list2], k=60, top_n=2)
        weighted = weighted_rrf([list1, list2], weights=[1.0, 1.0], k=60, top_n=2)

        # Results should match
        assert len(standard) == len(weighted)
        for i in range(len(standard)):
            assert standard[i][0].page_content == weighted[i][0].page_content

    def test_different_weights(self, sample_docs):
        """Different weights should affect ranking."""
        # Doc A in list1, Doc B in list2
        list1 = [(sample_docs[0], 0.9)]
        list2 = [(sample_docs[1], 0.9)]

        # Give more weight to list2
        weighted = weighted_rrf([list1, list2], weights=[0.1, 0.9], k=60, top_n=2)

        # Doc B should be first due to higher weight
        assert weighted[0][0].page_content == sample_docs[1].page_content

    def test_default_weights(self, sample_docs):
        """Should use equal weights when not specified."""
        list1 = [(sample_docs[0], 0.9)]
        list2 = [(sample_docs[1], 0.8)]

        result = weighted_rrf([list1, list2], weights=None, k=60, top_n=2)

        assert len(result) == 2

    def test_weight_count_mismatch_raises(self, sample_docs):
        """Should raise error when weight count doesn't match list count."""
        list1 = [(sample_docs[0], 0.9)]
        list2 = [(sample_docs[1], 0.8)]

        with pytest.raises(ValueError, match="Number of weights"):
            weighted_rrf([list1, list2], weights=[1.0], k=60, top_n=2)

    def test_zero_weight(self, sample_docs):
        """Zero weight should effectively ignore that list."""
        list1 = [(sample_docs[0], 0.9)]
        list2 = [(sample_docs[1], 0.9)]

        weighted = weighted_rrf([list1, list2], weights=[1.0, 0.0], k=60, top_n=2)

        # Only Doc A should have a score > 0
        assert weighted[0][0].page_content == sample_docs[0].page_content
        if len(weighted) > 1:
            assert weighted[1][1] == 0.0

    def test_empty_input(self):
        """Should return empty list for empty input."""
        result = weighted_rrf([], weights=[], k=60, top_n=5)
        assert result == []
