from justatom.processing.mask import IProcessor


def test_empty_prefix_does_not_add_leading_whitespace():
    assert IProcessor.do_prefix(None, "  passage  ", "") == "passage"


def test_non_empty_prefix_has_one_separator():
    assert IProcessor.do_prefix(None, "  passage  ", "  query:  ") == "query: passage"
