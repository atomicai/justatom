from justatom.configuring.prime import _build_config_tree


def test_build_config_tree_applies_defaults():
    cfg = _build_config_tree({})

    assert cfg.loguru["LOG_ROTATION"] == "10 MB"
    assert cfg.train.index_name == "justatom"
    assert cfg.train.do_scale_unit == 1
    assert cfg.train.model.props.dropout == 0.1


def test_build_config_tree_preserves_loguru_and_training_overrides():
    cfg = _build_config_tree(
        {
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

    assert cfg.loguru["LOG_FILE_NAME"] == "custom.log"
    assert cfg.train.model.props.dropout == 0.2
