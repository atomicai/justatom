# Habr IR Batch Cache Control

## Goal

Generate the next disjoint 3,000 Habr IR requests without GPT-5.6's implicit
prompt-cache writes, which produced no cache hits in the previous one-shot
Batch runs.

## Design

Add `generation.prompt_cache_mode` with two supported values:

- `auto` preserves the OpenAI API default and omits `prompt_cache_options`.
- `explicit` sends `prompt_cache_options: {mode: explicit}` without cache
  breakpoints, disabling the model's implicit breakpoint.

The default remains `auto` so existing configs and generation artifacts retain
their behavior. The selected mode participates in the normalized generation
config and generation fingerprint, preventing artifacts created under different
cache policies from being reused as if they were equivalent.

The next 3k config uses `explicit` and excludes both prior target roots. It
selects 1,500 new articles with two target slots each, then emits three Batch
shards of 1,000 requests.

## Validation

- Unit-test config validation and both request payload modes.
- Verify changing the mode changes the generation fingerprint.
- Verify the new target cohort has zero article and passage overlap with the
  prior 1k and 3k cohorts.
- Inspect every generated request for `prompt_cache_options.mode=explicit`.
- Submit only after all three shards contain exactly 1,000 unique requests.

## Scope

This change does not introduce shared cache keys or explicit breakpoints. Those
are unsuitable for the current requests because most prompt content is unique
and Batch workers may execute concurrently before a reusable prefix is warm.
