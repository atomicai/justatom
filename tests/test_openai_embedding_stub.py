import importlib.util
from pathlib import Path


def _stub_module():
    path = Path("tests/fixtures/openai_embedding_stub.py")
    spec = importlib.util.spec_from_file_location("openai_embedding_stub_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_vectors_are_stable_three_dimensional_topic_basis_vectors():
    stub = _stub_module()

    assert stub._vector("банк негативов") == [1.0, 0.0, 0.0]
    assert stub._vector("Qwen эмбеддинги") == [0.0, 1.0, 0.0]
    assert stub._vector("Weaviate хранит векторы") == [0.0, 0.0, 1.0]
