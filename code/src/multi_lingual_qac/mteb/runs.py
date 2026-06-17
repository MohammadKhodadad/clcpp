"""Run identity and structured report folders for repeated (e.g. daily) benchmarks.

Each benchmark run gets a timestamp-based id (optionally suffixed with a label) and
a self-contained folder ``reports/runs/<run_id>/`` holding the summary, comparison
tables, predictions, question analysis, and a ``run_metadata.json`` capturing what
produced it (dataset + sizes, models, git commit, host). A rolling
``reports/runs/index.csv`` and a ``reports/runs/latest`` pointer make day-to-day
trends easy to track.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from src.multi_lingual_qac.mteb.evaluation import (
    _dataset_config_name,
    _slugify,
)

# Key metrics tracked in the rolling index (one row per run x model).
INDEX_METRICS = ["main_score", "recall_at_10", "recall_at_100", "ndcg_at_10", "map_at_10", "mrr_at_10"]
INDEX_COLUMNS = [
    "run_id", "created_at", "dataset_repo", "dataset_variant", "git_commit",
    "n_queries", "n_corpus", "model",
] + INDEX_METRICS


def make_run_id(label: str | None = None, *, now: datetime | None = None) -> str:
    """``20260601-143052`` or ``20260601-143052_<label>`` (UTC, sortable)."""
    now = now or datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d-%H%M%S")
    if label:
        slug = _slugify(label)
        if slug and slug != "default":
            run_id = f"{run_id}_{slug}"
    return run_id


def git_info(project_root: Path) -> tuple[str | None, bool]:
    """Return (short commit, dirty?) for the repo, or (None, False) if unavailable."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return commit, bool(status)


def dataset_sizes(dataset_repo: str, revision: str, dataset_variant: str) -> dict[str, int]:
    """Row counts for queries/corpus/qrels (content fingerprint of the run)."""
    sizes: dict[str, int] = {}
    for base in ("queries", "corpus", "qrels"):
        try:
            config = _dataset_config_name(dataset_repo, revision, dataset_variant, base)
            sizes[base] = load_dataset(dataset_repo, config, split="train", revision=revision).num_rows
        except Exception:
            sizes[base] = -1
    return sizes


def write_run_metadata(
    run_dir: Path,
    *,
    run_id: str,
    created_at: str,
    dataset_repo: str,
    dataset_variant: str,
    dataset_revision: str,
    models: Iterable[str],
    batch_size: int,
    summaries: list,
    sizes: dict[str, int],
    git_commit: str | None,
    git_dirty: bool,
    corpus_repo: str = "",
) -> Path:
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "dataset_repo": dataset_repo,
        "corpus_repo": corpus_repo or dataset_repo,
        "dataset_variant": dataset_variant,
        "dataset_revision": dataset_revision,
        "dataset_sizes": sizes,
        "models": list(models),
        "batch_size": batch_size,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scores": {
            item.model_name: {
                "main_score": item.main_score,
                **{metric: item.metrics.get(metric) for metric in INDEX_METRICS if metric != "main_score"},
            }
            for item in summaries
        },
    }
    path = Path(run_dir) / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def append_index(
    index_path: Path,
    *,
    run_id: str,
    created_at: str,
    dataset_repo: str,
    dataset_variant: str,
    git_commit: str | None,
    sizes: dict[str, int],
    summaries: list,
) -> Path:
    """Append one row per model to the rolling trend log (creates header once)."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(INDEX_COLUMNS)
        for item in summaries:
            writer.writerow([
                run_id, created_at, dataset_repo, dataset_variant, git_commit or "",
                sizes.get("queries", -1), sizes.get("corpus", -1), item.model_name,
                *[round(item.metrics.get(m, item.main_score if m == "main_score" else float("nan")), 5)
                  if (m == "main_score" or m in item.metrics) else ""
                  for m in INDEX_METRICS],
            ])
    return index_path


def update_latest_pointer(runs_root: Path, run_id: str) -> None:
    """Point ``runs_root/latest`` at the given run id (symlink, or latest.txt fallback)."""
    runs_root = Path(runs_root)
    link = runs_root / "latest"
    try:
        if link.is_symlink() or (link.exists() and not link.is_dir()):
            link.unlink()
        if not (link.exists() and link.is_dir()):
            link.symlink_to(run_id)  # relative target inside runs_root
    except OSError:
        (runs_root / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
