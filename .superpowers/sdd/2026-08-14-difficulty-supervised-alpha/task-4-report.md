# Task 4 Report: Migrate Checkpoints and Record Provenance

## Implementation

- Research checkpoints now write `schema_version: 2`.
- The checkpoint loader accepts schemas `1` and `2`. Schema-v1 migration copies
  the top-level config and its `objective`, `alpha_gate`, and `experiment`
  mappings before removing retired fields or changing values, leaving the
  loaded payload unchanged.
- Schema-v1 migration removes `objective.pairwise_margin`, maps
  `alpha_gate.mix_weight` to `alpha_gate.supervision_weight`, and removes
  `mix_weight_warmup_steps` and `entropy_weight`.
- Every schema-v1 `atom_gate` checkpoint and every schema-v1 decoupled
  checkpoint is labeled `experiment.role: ablation`. Coupled schema-v1
  `vanilla` and `atomic` checkpoints retain their canonical roles.
- Atom-gate manifests now record the detached positive-softmax confidence
  target and detached alpha-head input gradient in `objective_contract`.
- Training documentation now states the confidence-supervised alpha equation
  and no longer describes lexical pair supervision.
- The checkpoint loader already had no `lexical_lookup` parameter or call site;
  a source scan confirms no remaining production-training or test reference.

## RED

Command:

```text
conda run -n justatom python -m pytest tests/test_training_module.py::test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation tests/test_training_job.py -q
```

Output:

```text
ERROR conda.cli.main_run:execute(148): `conda run python -m pytest tests/test_training_module.py::test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation tests/test_training_job.py -q` failed. (See above for error)
F.F....                                                                  [100%]
=================================== FAILURES ===================================
_ test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation _

tmp_path = PosixPath('/private/var/folders/0f/42v0qlvn4990mf303jyt874w0000gn/T/pytest-of-thebat/pytest-464/test_load_schema_v1_checkpoint0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x323ce1280>

    def test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation(tmp_path, monkeypatch):
        config = canonical_method_config(TrainingMethod.ATOM_GATE)
        original = ContrastiveTrainingModule.build(TinyEncoder(), config)
        historical_config = train_config_to_dict(config)
        historical_config["objective"].update(decoupled=True, pairwise_margin=0.2)
        historical_config["alpha_gate"].pop("supervision_weight")
        historical_config["alpha_gate"].update(
            mix_weight=0.7,
            mix_weight_warmup_steps=10,
            entropy_weight=0.1,
        )
        checkpoint = tmp_path / "historical-checkpoint.pt"
        payload = {
            "schema_version": 1,
            "resolved_config": historical_config,
            "state_dict": original.state_dict(),
            "optimizer_states": [],
            "epoch": 0,
            "global_step": 1,
        }
        monkeypatch.setattr("justatom.training.module.torch.load", lambda *_args, **_kwargs: payload)

>       restored, optimizer_states = ContrastiveTrainingModule.load_research_checkpoint(
            checkpoint,
            encoder=TinyEncoder(),
        )

tests/test_training_module.py:104:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
justatom/training/module.py:365: in load_research_checkpoint
    config = parse_train_config(resolved_config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
justatom/training/config.py:425: in parse_train_config
    config = _overlay_dataclass(base, payload)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
justatom/training/config.py:240: in _overlay_dataclass
    updates[name] = _overlay_dataclass(existing, value, field_path)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

current = ObjectiveConfig(temperature=0.05, learnable_temperature=True, decoupled=False, simcse_dropout_weight=0.1, soft_fn_attract_weight=0.0, soft_fn_topk=1)
raw = {'decoupled': True, 'learnable_temperature': True, 'pairwise_margin': 0.2, 'simcse_dropout_weight': 0.1, ...}
path = 'objective'

    def _overlay_dataclass(current: Any, raw: Mapping[str, Any], path: str = "") -> Any:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path or 'config'} must be a mapping")

        known = {item.name for item in fields(current)}
        unknown = set(raw) - known
        if unknown:
            name = sorted(unknown)[0]
>           raise ValueError(f"unknown configuration field: {_path(path, name)}")
E           ValueError: unknown configuration field: objective.pairwise_margin

justatom/training/config.py:233: ValueError
__________ test_atom_gate_manifest_records_detached_auxiliary_control __________

    def test_atom_gate_manifest_records_detached_auxiliary_control():
        manifest = RunManifest.from_config(
            canonical_method_config(TrainingMethod.ATOM_GATE),
            git_commit="abc123",
            git_dirty=False,
        )

>       assert manifest.objective_contract == {
            "contrastive_kernel": "coupled_infonce",
            "alpha_aux_gradient": "detached",
            "alpha_target": "detached_positive_softmax_confidence",
            "alpha_head_input_gradient": "detached",
        }
E       AssertionError: assert {'alpha_aux_g...pled_infonce'} == {'alpha_aux_g...pled_infonce'}
E
E         Omitting 2 identical items, use -vv to show
E         Right contains 2 more items:
E         {'alpha_head_input_gradient': 'detached',
E          'alpha_target': 'detached_positive_softmax_confidence'}
E         Use -v to get more diff

tests/test_training_job.py:46: AssertionError
=============================== warnings summary ===============================
../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/lightning_fabric/__init__.py:41
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/lightning_fabric/__init__.py:41: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.

../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/lightning_fabric/__init__.py:41
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/lightning_fabric/__init__.py:41: Deprecated call to `pkg_resources.declare_namespace('lightning_fabric')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages

../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_fontconfig_pattern.py:64
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_fontconfig_pattern.py:64: PyparsingDeprecationWarning: 'oneOf' deprecated - use 'one_of'
    prop = Group((name + Suppress("=") + comma_separated(value)) | oneOf(_CONSTANTS))

../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_fontconfig_pattern.py:85
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_fontconfig_pattern.py:85: PyparsingDeprecationWarning: 'parseString' deprecated - use 'parse_string'
    parser = parser.parseString(pattern)

../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_fontconfig_pattern.py:89
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_mathtext.py:45
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/matplotlib/_mathtext.py:45: PyparsingDeprecationWarning: 'enablePackrat' deprecated - use 'enable_packrat'
    ParserElement.enablePackrat()

../../../../miniconda3/envs/justatom/lib/python3.12/site-packages/pytorch_lightning/__init__.py:37
  /Users/thebat/miniconda3/envs/justatom/lib/python3.12/site-packages/pytorch_lightning/__init__.py:37: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_training_module.py::test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation
FAILED tests/test_training_job.py::test_atom_gate_manifest_records_detached_auxiliary_control
2 failed, 5 passed, 17 warnings in 5.50s
```

## GREEN

Migration and manifest command:

```text
conda run -n justatom python -m pytest tests/test_training_module.py::test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation tests/test_training_job.py -q
.......                                                                  [100%]
7 passed, 17 warnings in 5.48s
```

Focused verification:

```text
conda run -n justatom python -m pytest tests/test_training_module.py tests/test_training_job.py -q
........................                                                 [100%]
24 passed, 26 warnings in 5.74s

conda run -n justatom ruff check justatom/training/module.py justatom/training/job.py tests/test_training_module.py tests/test_training_job.py
All checks passed!

git diff --check
<no output; exit 0>

rg -n "lexical_lookup" justatom/training tests
<no output; exit 0>
```

The pytest warnings are pre-existing dependency and Lightning runtime warnings;
they do not report test failures.

## Files Changed

- `justatom/training/module.py`
- `justatom/training/job.py`
- `tests/test_training_module.py`
- `tests/test_training_job.py`
- `docs/training.md`
- `.superpowers/sdd/2026-08-14-difficulty-supervised-alpha/task-4-report.md`

## Self-Review

- Verified schema-v1 migration removes every listed retired field, maps the
  historical weight, does not mutate the loaded payload, and loads the alpha
  head state with `strict=True`.
- Added separate coverage for historical coupled `atom_gate` becoming an
  ablation and parameterized coverage that canonical coupled schema-v1
  `vanilla` and `atomic` remain canonical.
- Added schema-v2 writer coverage and verified the loader accepts schema 2 as
  part of the supported schema set.
- Confirmed only atom-gate manifests gain the new alpha provenance fields;
  non-gate manifest contracts remain unchanged.
- Confirmed the documentation formula matches the detached confidence target,
  detached head input, and detached SimCSE coefficient implementation.

## Concerns

None for the task implementation. The focused pytest commands emit existing
third-party deprecation and Lightning environment warnings, recorded above.
