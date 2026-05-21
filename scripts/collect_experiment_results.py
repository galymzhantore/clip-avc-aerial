"""Collect training/eval results from encoder sweep log directories.

Example:
    uv run python -m scripts.collect_experiment_results logs_encoder_sweep logs_overnight_encoder_search
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


ACC_RE = re.compile(r"(?P<dataset>\w+) (?P<split>\w+) accuracy: (?P<acc>[0-9.]+)% \((?P<correct>\d+)/(?P<total>\d+)\)")
CONFIG_RE = re.compile(r"config: (?P<config>\{.*\})")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _parse_config(text: str) -> dict:
    match = CONFIG_RE.search(text)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group("config"))
    except (SyntaxError, ValueError):
        return {}


def _parse_accuracy(text: str) -> dict:
    matches = list(ACC_RE.finditer(text))
    if not matches:
        return {}
    match = matches[-1]
    data = match.groupdict()
    data["acc"] = float(data["acc"])
    data["correct"] = int(data["correct"])
    data["total"] = int(data["total"])
    return data


def _parse_manifest(path: Path) -> dict:
    data = {}
    for line in _read_text(path).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def collect(log_roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in log_roots:
        if not root.exists():
            continue
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            train_text = _read_text(case_dir / "train_era_full.log")
            eval_text = _read_text(case_dir / "eval_era_full.log")
            config = _parse_config(train_text)
            acc = _parse_accuracy(eval_text)
            manifest = _parse_manifest(case_dir / "manifest.txt")
            rows.append(
                {
                    "log_root": str(root),
                    "case": case_dir.name,
                    "status": manifest.get("status", ""),
                    "started_at": manifest.get("started_at", ""),
                    "ended_at": manifest.get("ended_at", ""),
                    "accuracy": acc.get("acc", ""),
                    "correct": acc.get("correct", ""),
                    "total": acc.get("total", ""),
                    "video_model": config.get("video_model", ""),
                    "clip_model": config.get("clip_model", ""),
                    "text_encoder": config.get("text_encoder", ""),
                    "refined_text_pooling": config.get("refined_text_pooling", ""),
                    "classifier_mode": config.get("classifier_mode", ""),
                    "batch_size": config.get("batch_size", ""),
                    "micro_batch_size": config.get("micro_batch_size", ""),
                    "epochs": config.get("epochs", ""),
                    "log_dir": str(case_dir),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.log_roots)
    fieldnames = [
        "log_root",
        "case",
        "status",
        "started_at",
        "ended_at",
        "accuracy",
        "correct",
        "total",
        "video_model",
        "clip_model",
        "text_encoder",
        "refined_text_pooling",
        "classifier_mode",
        "batch_size",
        "micro_batch_size",
        "epochs",
        "log_dir",
    ]
    if args.out is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
