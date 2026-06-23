#!/usr/bin/env python3
"""Inject PersonaDrive (PCT) annotations into the NavSim *metric* cache.

Starting point: you have only downloaded OpenScene/NavSim and built the standard
metric cache, i.e. every evaluation token has a directory::

    <metric_cache>/<log_name>/unknown/<token>/
        metric_cache.pkl            # PDM scorer state (built by NavSim)
        gpt.json                    # persona annotations  (written by this script)

The PDM scorer itself needs nothing added -- it already lives in
``metric_cache.pkl``. What evaluation additionally needs is the persona text and
target trajectories per token. The main scorer (``run_pdm_score.py``) reads those
straight from ``per_scene_dir``, but the per-token visualisation / legacy paths
(e.g. ``run_pdm_score_vis.py``) read a ``gpt.json`` sitting next to
``metric_cache.pkl``. This script writes that ``gpt.json`` so both paths work
without rebuilding the scorer cache.

By default ``gpt.json`` is a verbatim copy of ``per_scene/<token>.json``. With
``--tokenize`` each persona entry additionally gets a pre-computed::

    scene[cat]["tokenized"][<bert_backbone>] = {
        "input_ids":      [.. max_length ints ..],
        "attention_mask": [.. max_length ints ..],
    }

which the scorer's fast-path picks up to skip on-the-fly tokenisation.

Tokens in the metric cache without a matching ``per_scene/<token>.json`` are left
untouched and reported as ``skipped``.

Example::

    python inject_to_metric_cache.py \
        --metric_cache_dir $NAVSIM_EXP_ROOT/metric_cache \
        --per_scene_dir    $NAVSIM_EXP_ROOT/personadrive/per_scene \
        --tokenize --num_workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Dict, List, Tuple

from tqdm import tqdm

import personadrive_common as pc

METRIC_CACHE_FILE = "metric_cache.pkl"
GPT_JSON_FILE = "gpt.json"

# Per-worker tokenizer (only built when --tokenize is set).
_TOKENIZER = None
_MAX_LENGTH = pc.DEFAULT_MAX_LENGTH
_BACKBONE = pc.DEFAULT_BERT_BACKBONE


def discover_token_dirs(metric_cache_dir: str) -> Dict[str, str]:
    """Map ``{token: token_dir}`` by locating every ``metric_cache.pkl``.

    Works regardless of the intermediate layout (``<log>/unknown/<token>/`` in
    NavSim) because it keys off the file name rather than a fixed depth.
    """
    token_to_dir: Dict[str, str] = {}
    for pkl in glob(os.path.join(metric_cache_dir, "**", METRIC_CACHE_FILE),
                    recursive=True):
        token_dir = os.path.dirname(pkl)
        token_to_dir[os.path.basename(token_dir)] = token_dir
    return token_to_dir


def inject_one(
    token: str,
    token_dir: str,
    per_scene_dir: str,
    tokenize: bool,
    tokenizer,
    backbone: str,
    max_length: int,
    dry_run: bool,
) -> str:
    """Write ``gpt.json`` for a single token. Return a status string.

    Status values: ``"injected"``, ``"no_json"``, or ``"error: <msg>"``.
    """
    if not os.path.exists(pc.per_scene_json_path(per_scene_dir, token)):
        return "no_json"

    try:
        scene = pc.load_per_scene(per_scene_dir, token)
        if tokenize:
            for cat in pc.CATEGORIES:
                encoded = pc.tokenize_persona_lists(
                    tokenizer, scene[cat]["user_intent"], max_length
                )
                scene[cat].setdefault("tokenized", {})[backbone] = encoded
    except Exception as e:
        return f"error: {e}"

    if dry_run:
        return "injected"

    with open(os.path.join(token_dir, GPT_JSON_FILE), "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)
    return "injected"


# --------------------------- multiprocessing plumbing ---------------------------

def _worker_init(backbone: str, max_length: int, tokenize: bool) -> None:
    """Build one tokenizer per worker process, only if tokenising."""
    global _TOKENIZER, _MAX_LENGTH, _BACKBONE
    _MAX_LENGTH = max_length
    _BACKBONE = backbone
    _TOKENIZER = pc.build_tokenizer(backbone) if tokenize else None


def _worker_inject(job: Tuple[str, str, str, bool, bool]) -> str:
    """Worker wrapper that reuses the process-local tokenizer."""
    token, token_dir, per_scene_dir, tokenize, dry_run = job
    return inject_one(
        token, token_dir, per_scene_dir,
        tokenize, _TOKENIZER, _BACKBONE, _MAX_LENGTH, dry_run,
    )


def _tally(statuses: List[str]) -> Dict[str, int]:
    """Collapse per-token statuses into counts (errors bucketed together)."""
    counts: Dict[str, int] = {}
    for s in statuses:
        key = "error" if s.startswith("error:") else s
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    exp_root = pc.default_exp_root()
    parser = argparse.ArgumentParser(
        description="Inject PersonaDrive annotations into the NavSim metric cache."
    )
    parser.add_argument(
        "--metric_cache_dir", type=str,
        default=os.path.join(exp_root, "metric_cache") if exp_root else None,
        required=not exp_root,
        help="Metric cache root (contains metric_cache.pkl per token). "
             "Default: $NAVSIM_EXP_ROOT/metric_cache",
    )
    parser.add_argument(
        "--per_scene_dir", type=str,
        default=os.path.join(exp_root, "personadrive", "per_scene") if exp_root else None,
        required=not exp_root,
        help="PersonaDrive per_scene/<token>.json directory. "
             "Default: $NAVSIM_EXP_ROOT/personadrive/per_scene",
    )
    parser.add_argument(
        "--tokenize", action="store_true",
        help="Also embed pre-tokenized input_ids/attention_mask per persona "
             "(keyed by --bert_backbone) so the scorer skips on-the-fly tokenisation.",
    )
    parser.add_argument(
        "--bert_backbone", type=str, default=pc.DEFAULT_BERT_BACKBONE,
        help="HuggingFace tokenizer used when --tokenize is set.",
    )
    parser.add_argument(
        "--max_length", type=int, default=pc.DEFAULT_MAX_LENGTH,
        help="Tokenizer max_length / padding length.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Process pool size (0 = single process).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate (and tokenise) without writing any gpt.json.",
    )
    args = parser.parse_args()

    print(f"Scanning metric cache: {args.metric_cache_dir}")
    token_to_dir = discover_token_dirs(args.metric_cache_dir)
    tokens = sorted(token_to_dir)
    print(f"  found {len(tokens)} token directories")
    print(f"PersonaDrive annotations: {args.per_scene_dir}")
    if args.tokenize:
        print(f"Embedding pre-tokenized text with backbone: {args.bert_backbone}")

    statuses: List[str] = []
    if args.num_workers and args.num_workers > 1:
        jobs = [
            (t, token_to_dir[t], args.per_scene_dir, args.tokenize, args.dry_run)
            for t in tokens
        ]
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(args.bert_backbone, args.max_length, args.tokenize),
        ) as ex:
            futures = [ex.submit(_worker_inject, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Injecting"):
                statuses.append(fut.result())
    else:
        tokenizer = pc.build_tokenizer(args.bert_backbone) if args.tokenize else None
        for t in tqdm(tokens, desc="Injecting"):
            statuses.append(inject_one(
                t, token_to_dir[t], args.per_scene_dir,
                args.tokenize, tokenizer, args.bert_backbone,
                args.max_length, args.dry_run,
            ))

    counts = _tally(statuses)
    print("\n=== Summary ===")
    print(f"Total cache tokens     : {len(tokens)}")
    print(f"Injected               : {counts.get('injected', 0)}")
    print(f"Skipped (no per_scene) : {counts.get('no_json', 0)}")
    print(f"Errors                 : {counts.get('error', 0)}")
    if counts.get("error"):
        print("First few errors:")
        shown = 0
        for s in statuses:
            if s.startswith("error:") and shown < 5:
                print(f"  {s}")
                shown += 1
    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")


if __name__ == "__main__":
    main()
