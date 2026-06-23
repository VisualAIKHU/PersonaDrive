#!/usr/bin/env python3
"""Inject PersonaDrive (PCT) annotations into the NavSim *training* cache.

Starting point: you have only downloaded OpenScene/NavSim and built the standard
DiffusionDrive feature cache, i.e. each scene token has a directory::

    <training_cache>/<log_name>/<token>/
        transfuser_feature.gz       # camera / lidar / status (+ text, after this)
        transfuser_target.gz        # GT trajectory + BEV labels (+ persona, after this)

This script layers the persona annotations from ``per_scene/<token>.json`` on top,
so the cache can train the persona-conditioned agent. For every token it:

    1. tokenises the 9 ``user_intent`` strings and writes, into
       ``transfuser_feature.gz``:
           input_ids       -> list of 9 tensors [max_length]
           attention_mask  -> list of 9 tensors [max_length]
    2. writes, into ``transfuser_target.gz``:
           trajectories    -> list of 9 tensors [8, 3]   (first 8 of 10 waypoints)
           categories      -> list of 9 one-hot tensors [9]

The agent reads ``input_ids`` / ``attention_mask`` from the feature file and
``trajectories`` / ``categories`` from the target file, so those are the exact
keys / files we populate.

Tokens present in the cache but missing a ``per_scene/<token>.json`` are left
untouched and reported as ``skipped`` -- restrict your training split to the
injected tokens so the dataloader never pulls an un-augmented sample.

Example::

    python inject_to_train_cache.py \
        --cache_dir   $NAVSIM_EXP_ROOT/training_cache \
        --per_scene_dir $NAVSIM_EXP_ROOT/personadrive/per_scene \
        --num_workers 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Dict, List, Tuple

from tqdm import tqdm

import personadrive_common as pc

FEATURE_FILE = "transfuser_feature.gz"
TARGET_FILE = "transfuser_target.gz"

# Per-worker tokenizer, created once per process by ``_worker_init``.
_TOKENIZER = None
_MAX_LENGTH = pc.DEFAULT_MAX_LENGTH


def discover_token_dirs(cache_dir: str) -> Dict[str, str]:
    """Map ``{token: token_dir}`` from the ``<log>/<token>/`` cache layout."""
    token_to_dir: Dict[str, str] = {}
    for token_dir in glob(os.path.join(cache_dir, "*", "*")):
        if os.path.isdir(token_dir):
            token_to_dir[os.path.basename(token_dir)] = token_dir
    return token_to_dir


def inject_one(
    token: str,
    token_dir: str,
    per_scene_dir: str,
    tokenizer,
    max_length: int,
    dry_run: bool,
) -> str:
    """Inject a single token. Return a status string for tallying.

    Status values: ``"injected"``, ``"no_json"``, ``"no_feature"``,
    ``"no_target"``, or ``"error: <msg>"``.
    """
    json_path = pc.per_scene_json_path(per_scene_dir, token)
    if not os.path.exists(json_path):
        return "no_json"

    feat_path = os.path.join(token_dir, FEATURE_FILE)
    if not os.path.exists(feat_path):
        return "no_feature"

    tgt_path = os.path.join(token_dir, TARGET_FILE)
    if not os.path.exists(tgt_path):
        return "no_target"

    try:
        scene = pc.load_per_scene(per_scene_dir, token)
        input_ids, attention_mask = pc.tokenize_personas(
            tokenizer, pc.extract_personas(scene), max_length
        )
        trajectories = pc.extract_trajectories(scene)
        categories = pc.one_hot_categories()
    except Exception as e:  # malformed annotation, missing key, etc.
        return f"error: {e}"

    if dry_run:
        return "injected"

    # --- feature file: add text tokens, drop the legacy precomputed-embedding key ---
    feature = pc.load_gz(feat_path)
    feature.pop("bert_emb", None)  # remove obsolete key from older pipelines
    feature["input_ids"] = input_ids
    feature["attention_mask"] = attention_mask
    pc.dump_gz(feat_path, feature)

    # --- target file: add per-persona trajectories + one-hot categories ---
    target = pc.load_gz(tgt_path)
    target["trajectories"] = trajectories
    target["categories"] = categories
    pc.dump_gz(tgt_path, target)

    return "injected"


# --------------------------- multiprocessing plumbing ---------------------------

def _worker_init(backbone: str, max_length: int) -> None:
    """Build one tokenizer per worker process (tokenizers are not fork-safe)."""
    global _TOKENIZER, _MAX_LENGTH
    _TOKENIZER = pc.build_tokenizer(backbone)
    _MAX_LENGTH = max_length


def _worker_inject(job: Tuple[str, str, str, bool]) -> str:
    """Worker wrapper that reuses the process-local tokenizer."""
    token, token_dir, per_scene_dir, dry_run = job
    return inject_one(
        token, token_dir, per_scene_dir,
        _TOKENIZER, _MAX_LENGTH, dry_run,
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
        description="Inject PersonaDrive annotations into the NavSim training cache."
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default=os.path.join(exp_root, "training_cache") if exp_root else None,
        required=not exp_root,
        help="Training cache root (<log>/<token>/ layout). "
             "Default: $NAVSIM_EXP_ROOT/training_cache",
    )
    parser.add_argument(
        "--per_scene_dir", type=str,
        default=os.path.join(exp_root, "personadrive", "per_scene") if exp_root else None,
        required=not exp_root,
        help="PersonaDrive per_scene/<token>.json directory. "
             "Default: $NAVSIM_EXP_ROOT/personadrive/per_scene",
    )
    parser.add_argument(
        "--bert_backbone", type=str, default=pc.DEFAULT_BERT_BACKBONE,
        help="HuggingFace tokenizer to encode user_intent.",
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
        help="Validate and tokenise without writing any .gz file.",
    )
    args = parser.parse_args()

    print(f"Scanning training cache: {args.cache_dir}")
    token_to_dir = discover_token_dirs(args.cache_dir)
    tokens = sorted(token_to_dir)
    print(f"  found {len(tokens)} token directories")
    print(f"PersonaDrive annotations: {args.per_scene_dir}")

    statuses: List[str] = []
    if args.num_workers and args.num_workers > 1:
        jobs = [
            (t, token_to_dir[t], args.per_scene_dir, args.dry_run)
            for t in tokens
        ]
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(args.bert_backbone, args.max_length),
        ) as ex:
            futures = [ex.submit(_worker_inject, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Injecting"):
                statuses.append(fut.result())
    else:
        tokenizer = pc.build_tokenizer(args.bert_backbone)
        for t in tqdm(tokens, desc="Injecting"):
            statuses.append(inject_one(
                t, token_to_dir[t], args.per_scene_dir,
                tokenizer, args.max_length, args.dry_run,
            ))

    counts = _tally(statuses)
    print("\n=== Summary ===")
    print(f"Total cache tokens     : {len(tokens)}")
    print(f"Injected               : {counts.get('injected', 0)}")
    print(f"Skipped (no per_scene) : {counts.get('no_json', 0)}")
    print(f"Skipped (no feature.gz): {counts.get('no_feature', 0)}")
    print(f"Skipped (no target.gz) : {counts.get('no_target', 0)}")
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
