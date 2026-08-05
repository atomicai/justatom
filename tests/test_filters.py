from unittest.mock import patch

import polars as pl

from justatom.etc import filters


def test_filter_value_serializes_polars_dataframe_without_pandas():
    captured = {}

    def fake_equal(field, value):
        captured["field"] = field
        captured["value"] = value
        return "converted"

    with patch.dict(filters.COMPARISON_OPERATORS, {"==": fake_equal}):
        result = filters._parse_comparison_condition(
            {
                "field": "table",
                "operator": "==",
                "value": pl.DataFrame({"answer": [42]}),
            }
        )

    assert result == "converted"
    assert captured["field"] == "table"
    assert isinstance(captured["value"], str)
    assert "answer" in captured["value"]
