from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _slugify(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "run"


@dataclass
class _LossPoint:
    split: str
    step: int
    wallclock_seconds: float
    loss: float


class ExperimentTracker:
    CSV_FIELDS = [
        "event",
        "split",
        "step",
        "wallclock_seconds",
        "loss",
        "ppl",
        "lr",
        "tokens_seen",
        "step_time_seconds",
        "eval_time_seconds",
        "checkpoint_path",
        "message",
    ]

    def __init__(
        self,
        root_dir: str | Path,
        *,
        config: dict[str, Any],
        run_name: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base_name = _slugify(run_name) if run_name else f"run-{timestamp}"
        self.run_dir = self._allocate_run_dir(base_name)
        self.run_name = self.run_dir.name

        self.metrics_jsonl_path = self.run_dir / "metrics.jsonl"
        self.metrics_csv_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"

        self._loss_points: list[_LossPoint] = []
        self._num_events = 0
        self._best_val_loss: float | None = None
        self._best_val_step: int | None = None
        self._last_train_loss: float | None = None
        self._last_val_loss: float | None = None

        self._write_json(
            self.config_path,
            {
                "run_name": self.run_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": _json_safe(config),
            },
        )
        self._write_csv_header()

    def _allocate_run_dir(self, base_name: str) -> Path:
        candidate = self.root_dir / base_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

        suffix = 1
        while True:
            candidate = self.root_dir / f"{base_name}-{suffix:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            suffix += 1

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_csv_header(self) -> None:
        with self.metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

    def _append_event(self, event: dict[str, Any]) -> None:
        json_ready = _json_safe(event)
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready, sort_keys=True) + "\n")

        csv_row = {field: "" for field in self.CSV_FIELDS}
        for field in self.CSV_FIELDS:
            if field in json_ready:
                csv_row[field] = json_ready[field]
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writerow(csv_row)

        self._num_events += 1

    def log_metrics(self, *, split: str, step: int, wallclock_seconds: float, **metrics: Any) -> None:
        event = {
            "event": "metric",
            "split": split,
            "step": int(step),
            "wallclock_seconds": float(wallclock_seconds),
            **metrics,
        }
        self._append_event(event)

        loss = metrics.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            point = _LossPoint(
                split=split,
                step=int(step),
                wallclock_seconds=float(wallclock_seconds),
                loss=float(loss),
            )
            self._loss_points.append(point)
            if split == "train":
                self._last_train_loss = float(loss)
            if split == "val":
                self._last_val_loss = float(loss)
                if self._best_val_loss is None or float(loss) < self._best_val_loss:
                    self._best_val_loss = float(loss)
                    self._best_val_step = int(step)

    def log_checkpoint(self, *, step: int, wallclock_seconds: float, checkpoint_path: str | Path) -> None:
        self._append_event(
            {
                "event": "checkpoint",
                "step": int(step),
                "wallclock_seconds": float(wallclock_seconds),
                "checkpoint_path": str(checkpoint_path),
            }
        )

    def finish(self, *, status: str, final_step: int) -> None:
        self._write_loss_svg(
            path=self.run_dir / "loss_vs_step.svg",
            title="Loss vs Gradient Step",
            x_attr="step",
            x_label="gradient step",
        )
        self._write_loss_svg(
            path=self.run_dir / "loss_vs_wallclock.svg",
            title="Loss vs Wallclock Time",
            x_attr="wallclock_seconds",
            x_label="wallclock seconds",
        )
        self._write_json(
            self.summary_path,
            {
                "run_name": self.run_name,
                "status": status,
                "final_step": int(final_step),
                "num_events": self._num_events,
                "last_train_loss": self._last_train_loss,
                "last_val_loss": self._last_val_loss,
                "best_val_loss": self._best_val_loss,
                "best_val_step": self._best_val_step,
                "artifacts": {
                    "config": str(self.config_path),
                    "metrics_jsonl": str(self.metrics_jsonl_path),
                    "metrics_csv": str(self.metrics_csv_path),
                    "loss_vs_step_svg": str(self.run_dir / "loss_vs_step.svg"),
                    "loss_vs_wallclock_svg": str(self.run_dir / "loss_vs_wallclock.svg"),
                },
            },
        )

    def _write_loss_svg(self, *, path: Path, title: str, x_attr: str, x_label: str) -> None:
        points_by_split = {
            "train": [point for point in self._loss_points if point.split == "train"],
            "val": [point for point in self._loss_points if point.split == "val"],
        }
        all_points = points_by_split["train"] + points_by_split["val"]
        width, height = 800, 420
        left, right, top, bottom = 70, 30, 50, 55
        plot_width = width - left - right
        plot_height = height - top - bottom

        if not all_points:
            empty_svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="white"/>'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                f'font-family="monospace" font-size="16">No loss points logged</text></svg>'
            )
            path.write_text(empty_svg, encoding="utf-8")
            return

        xs = [float(getattr(point, x_attr)) for point in all_points]
        ys = [point.loss for point in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if math.isclose(min_x, max_x):
            max_x = min_x + 1.0
        if math.isclose(min_y, max_y):
            max_y = min_y + 1.0

        def scale_x(x_value: float) -> float:
            return left + (x_value - min_x) / (max_x - min_x) * plot_width

        def scale_y(y_value: float) -> float:
            return top + (max_y - y_value) / (max_y - min_y) * plot_height

        colors = {"train": "#2563eb", "val": "#dc2626"}
        labels = {"train": "train", "val": "val"}
        line_parts: list[str] = []
        legend_parts: list[str] = []
        legend_y = top
        for split in ("train", "val"):
            split_points = points_by_split[split]
            if not split_points:
                continue
            polyline_points = " ".join(
                f"{scale_x(float(getattr(point, x_attr))):.1f},{scale_y(point.loss):.1f}"
                for point in split_points
            )
            line_parts.append(
                f'<polyline fill="none" stroke="{colors[split]}" stroke-width="2" points="{polyline_points}"/>'
            )
            legend_parts.append(
                f'<rect x="{width - 150}" y="{legend_y - 10}" width="12" height="12" fill="{colors[split]}"/>'
                f'<text x="{width - 132}" y="{legend_y}" font-family="monospace" font-size="12">{labels[split]}</text>'
            )
            legend_y += 20

        x_ticks = [min_x + i * (max_x - min_x) / 4 for i in range(5)]
        y_ticks = [min_y + i * (max_y - min_y) / 4 for i in range(5)]
        x_tick_parts = []
        for tick in x_ticks:
            x_pos = scale_x(tick)
            x_tick_parts.append(
                f'<line x1="{x_pos:.1f}" y1="{height - bottom}" x2="{x_pos:.1f}" y2="{height - bottom + 6}" stroke="black"/>'
                f'<text x="{x_pos:.1f}" y="{height - bottom + 22}" text-anchor="middle" font-family="monospace" '
                f'font-size="11">{tick:.1f}</text>'
            )
        y_tick_parts = []
        for tick in y_ticks:
            y_pos = scale_y(tick)
            y_tick_parts.append(
                f'<line x1="{left - 6}" y1="{y_pos:.1f}" x2="{left}" y2="{y_pos:.1f}" stroke="black"/>'
                f'<text x="{left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" font-family="monospace" '
                f'font-size="11">{tick:.3f}</text>'
            )

        svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="monospace" font-size="18">{title}</text>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>
  {''.join(x_tick_parts)}
  {''.join(y_tick_parts)}
  {''.join(line_parts)}
  {''.join(legend_parts)}
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-family="monospace" font-size="12">{x_label}</text>
  <text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})"
        font-family="monospace" font-size="12">loss</text>
</svg>
"""
        path.write_text(svg.strip() + "\n", encoding="utf-8")
