import math

import pytest

from justatom.retrieval.contracts import EmbeddingProfile, SearchMode, apply_prefix, validate_embeddings
from justatom.retrieval.errors import ConfigurationError, EmbeddingResponseError


def test_profile_validates_positive_limits_and_avoids_double_prefix():
    profile = EmbeddingProfile(query_prefix="query: ", document_prefix="passage: ")
    assert apply_prefix("cats", profile.query_prefix, skip_if_present=True) == "query: cats"
    assert apply_prefix("query: cats", profile.query_prefix, skip_if_present=True) == "query: cats"
    with pytest.raises(ConfigurationError, match="batch_size"):
        EmbeddingProfile(batch_size=0)


def test_validate_embeddings_checks_count_dimension_and_finite_values():
    assert validate_embeddings([[1, 2], [3.5, 4]], expected_count=2) == [[1.0, 2.0], [3.5, 4.0]]
    with pytest.raises(EmbeddingResponseError, match="Expected 2 vectors"):
        validate_embeddings([[1.0]], expected_count=2)
    with pytest.raises(EmbeddingResponseError, match="dimension"):
        validate_embeddings([[1.0], [1.0, 2.0]], expected_count=2)
    with pytest.raises(EmbeddingResponseError, match="finite"):
        validate_embeddings([[math.nan]], expected_count=1)


def test_search_modes_are_only_keyword_vector_and_hybrid():
    assert {mode.value for mode in SearchMode} == {"keyword", "vector", "hybrid"}
