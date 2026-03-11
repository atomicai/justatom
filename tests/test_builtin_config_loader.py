from justatom.configuring.prime import _build_config_tree


def test_build_config_tree_applies_defaults():
    cfg = _build_config_tree({})

    assert cfg.loguru["LOG_ROTATION"] == "10 MB"
    assert cfg.api.gpu_props.devices == ["cuda", "mps", "cpu"]
    assert cfg.train.index_name == "justatom"
    assert cfg.train.do_scale_unit == 1
    assert cfg.train.model.props.dropout == 0.1


def test_build_config_tree_normalizes_legacy_api_keys():
    cfg = _build_config_tree(
        {
            "api": {
                "devices_to_use": ["cuda:0", "cpu"],
                "embedder": {
                    "model_name_or_path": "legacy-model",
                    "max_seq_len": 256,
                },
            },
            "loguru": {
                "LOG_FILE_NAME": "custom.log",
            },
            "train": {
                "model": {
                    "props": {
                        "dropout": 0.2,
                    },
                },
            },
        }
    )

    assert cfg.api.model_name_or_path == "legacy-model"
    assert cfg.api.max_seq_len == 256
    assert cfg.api.gpu_props.devices == ["cuda:0", "cpu"]
    assert cfg.loguru["LOG_FILE_NAME"] == "custom.log"
    assert cfg.train.model.props.dropout == 0.2
