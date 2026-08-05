from unittest.mock import patch

from justatom.api import datasets as datasets_api


def test_default_prompt_paths_use_packaged_resources():
    kwargs = datasets_api.resolve_datasets_kwargs(
        config={
            "dataset": {"name_or_path": "demo"},
            "prompt": {"system_path": None, "user_template_path": None},
        }
    )

    assert "search queries" in kwargs["system_prompt"].lower()
    assert "{{content}}" in kwargs["user_template"]


def test_custom_prompt_path_is_a_normal_file(tmp_path):
    system_path = tmp_path / "system.txt"
    template_path = tmp_path / "template.txt"
    system_path.write_text("custom system", encoding="utf-8")
    template_path.write_text("custom {{ content }}", encoding="utf-8")

    kwargs = datasets_api.resolve_datasets_kwargs(
        config={
            "dataset": {"name_or_path": "demo"},
            "prompt": {
                "system_path": str(system_path),
                "user_template_path": str(template_path),
            },
        }
    )

    assert kwargs["system_prompt"] == "custom system"
    assert kwargs["user_template"] == "custom {{ content }}"


def test_rows_from_source_forwards_shared_loader_options():
    expected = [{"content": "one"}]

    with patch.object(datasets_api.DatasetLoader, "read", return_value=iter(expected)) as mocked:
        rows = datasets_api._rows_from_source(
            "owner/data",
            lazy=True,
            config="russian",
            split="train",
            limit=1,
            drop_columns=["blob"],
        )

        assert list(rows) == expected

    mocked.assert_called_once_with(
        "owner/data",
        lazy=True,
        config="russian",
        split="train",
        limit=1,
        drop_columns=["blob"],
    )
