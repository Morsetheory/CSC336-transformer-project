"""Regression tests for the production training and decoding paths."""

import json
import sys

import numpy as np
import pytest
import torch

import train_together
from cs336_basics.decoder import _sample_from_logits


@pytest.mark.parametrize("length", [4, 5])
def test_get_batch_includes_last_valid_window(length):
    data = np.arange(length, dtype=np.uint16)
    torch.manual_seed(0)
    inputs, targets = train_together.get_batch(data, 64, 3, "cpu")

    assert inputs.shape == targets.shape == (64, 3)
    assert inputs.dtype == targets.dtype == torch.long
    assert set(inputs[:, 0].tolist()) == set(range(length - 3))
    torch.testing.assert_close(targets, inputs + 1)


@pytest.mark.parametrize(
    "probabilities,top_p,expected",
    [
        ([0.6, 0.3, 0.1], 0.8, [2 / 3, 1 / 3, 0.0]),
        ([0.1, 0.6, 0.3], 0.8, [0.0, 2 / 3, 1 / 3]),
        ([0.5, 0.25, 0.25], 0.5, [1.0, 0.0, 0.0]),
        ([0.6, 0.3, 0.1], 0.1, [1.0, 0.0, 0.0]),
        ([0.6, 0.3, 0.1], 1.0, [0.6, 0.3, 0.1]),
    ],
)
def test_top_p_sampling_distribution(monkeypatch, probabilities, top_p, expected):
    captured = []

    def capture_multinomial(probs, num_samples):
        captured.append(probs)
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    logits = torch.tensor([probabilities], dtype=torch.float64).log()
    _sample_from_logits(logits, temperature=1.0, top_p=top_p)
    torch.testing.assert_close(captured[0], torch.tensor([expected], dtype=torch.float64))


def test_missing_resume_fails_before_preparing_data(tmp_path, monkeypatch):
    checkpoint = tmp_path / "missing.pt"
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_together.py", "--resume", str(checkpoint),
            "--text", "a small training example", "--prepared-dir", str(prepared),
            "--device", "cpu",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        train_together.main()
    assert not prepared.exists()


def test_training_and_resume_with_one_valid_window(tmp_path, monkeypatch):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    np.arange(4, dtype=np.uint16).tofile(train_path)
    np.arange(4, dtype=np.uint16).tofile(val_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    experiments = tmp_path / "experiments"
    argv = [
        "train_together.py", "--train-bin", str(train_path), "--val-bin", str(val_path),
        "--checkpoint-path", str(checkpoint_path), "--experiment-dir", str(experiments),
        "--device", "cpu", "--batch-size", "1", "--seq-len", "3",
        "--d-model", "8", "--num-layers", "1", "--num-heads", "2", "--d-ff", "16",
        "--eval-every", "1", "--eval-batches", "1", "--log-every", "1", "--save-every", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--steps", "1", "--run-name", "initial"])
    train_together.main()
    initial = torch.load(checkpoint_path, map_location="cpu")
    assert initial["iteration"] == 1
    assert initial["optimizer_state_dict"]["state"]
    summary = json.loads((experiments / "initial" / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert np.isfinite(summary["last_val_loss"])

    monkeypatch.setattr(
        sys, "argv",
        argv + ["--steps", "2", "--run-name", "resumed", "--resume", str(checkpoint_path)],
    )
    train_together.main()
    resumed = torch.load(checkpoint_path, map_location="cpu")
    assert resumed["iteration"] == 2
    assert all(state["t"] == 2 for state in resumed["optimizer_state_dict"]["state"].values())
    config = json.loads((experiments / "resumed" / "config.json").read_text())
    assert config["config"]["start_step"] == 1
    assert all(torch.isfinite(value).all() for value in resumed["model_state_dict"].values())
