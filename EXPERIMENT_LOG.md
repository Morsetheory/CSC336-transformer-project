# Experiment Log

This document records experiments run for the `train_together.py` training loop after adding local experiment tracking.

## Tracking Infrastructure

Training runs now write artifacts under `artifacts/experiments/<run-name>/`:

- `config.json`: resolved run configuration
- `metrics.jsonl`: append-only event log
- `metrics.csv`: tabular metrics for quick inspection
- `summary.json`: final status and best/last losses
- `loss_vs_step.svg`: train/val loss curve against gradient step
- `loss_vs_wallclock.svg`: train/val loss curve against wallclock seconds

Tracked fields include:

- gradient step
- wallclock seconds since training start
- train loss / perplexity / learning rate
- validation loss / perplexity
- tokens seen
- per-step and per-eval timing
- checkpoint save events

## Runs

### 2026-03-11: `tracking-smoke`

- Goal: first end-to-end smoke test of the new tracking pipeline.
- Command:

```bash
.venv/bin/python train_together.py \
  --text-file data/TinyStoriesV2-GPT4-valid.txt \
  --prepared-dir artifacts/tracking_smoke_data \
  --experiment-dir artifacts/experiments \
  --run-name tracking-smoke \
  --checkpoint-path artifacts/experiments/tracking-smoke/checkpoint.pt \
  --steps 3 \
  --batch-size 2 \
  --seq-len 16 \
  --eval-every 1 \
  --eval-batches 1 \
  --log-every 1 \
  --save-every 3 \
  --device cpu
```

- Result: failed before step 1.
- Failure: `NameError: name 'run_softmax' is not defined` from `cs336_basics/layers.py` inside `run_scaled_dot_product_attention()`.
- Artifacts:
  - [summary.json](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke/summary.json)
  - [metrics.csv](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke/metrics.csv)
- Outcome: fixed the bug by changing the attention code to call the local `softmax()` helper.

### 2026-03-11: `tracking-smoke-01`

- Goal: verify that experiment tracking works end to end after fixing the attention bug.
- Command:

```bash
.venv/bin/python train_together.py \
  --text-file data/TinyStoriesV2-GPT4-valid.txt \
  --prepared-dir artifacts/tracking_smoke_data \
  --experiment-dir artifacts/experiments \
  --run-name tracking-smoke \
  --checkpoint-path artifacts/experiments/tracking-smoke/checkpoint.pt \
  --steps 3 \
  --batch-size 2 \
  --seq-len 16 \
  --eval-every 1 \
  --eval-batches 1 \
  --log-every 1 \
  --save-every 3 \
  --device cpu
```

- Result: completed successfully.
- Final train loss: `6.4564`
- Final val loss: `5.8700`
- Best val loss: `5.8700` at step `3`
- Logged events: `7`
- Artifacts:
  - [config.json](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke-01/config.json)
  - [metrics.csv](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke-01/metrics.csv)
  - [summary.json](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke-01/summary.json)
  - [loss_vs_step.svg](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke-01/loss_vs_step.svg)
  - [loss_vs_wallclock.svg](/Users/shaoguanhua/Desktop/assignment1-basics/artifacts/experiments/tracking-smoke-01/loss_vs_wallclock.svg)
- Notes:
  - The run was intentionally tiny and is only a verification pass for logging.
  - Use the same logging outputs for the assignment’s real hyperparameter experiments.

## Suggested Workflow For Future Experiments

1. Choose a descriptive `--run-name`.
2. Keep `--experiment-dir artifacts/experiments`.
3. Run training.
4. Record the command, goal, and summary metrics in this file.
5. Compare `loss_vs_step.svg` and `loss_vs_wallclock.svg` across runs.
