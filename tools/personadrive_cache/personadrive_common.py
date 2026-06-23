#!/usr/bin/env python3
"""Shared helpers for expanding an OpenScene/NavSim cache into PersonaDrive (PCT).

PersonaDrive ("Persona-Conditioned Trajectory", PCT) is a layer of persona
annotations on top of the public OpenScene-v1.1 / NavSim scenes. The annotation
for one scene lives in ``per_scene/<token>.json`` and is keyed by the 9 persona
categories below. Each persona entry holds:

    {
        "trajectory":  [[x, y, heading], ... 10 future waypoints ...],
        "params":      {"maneuver": ..., "speed_factor": ..., "lane_offset": ...},
        "user_intent": "<natural-language passenger utterance>"
    }

This module centralises everything the two injection scripts share:
the fixed category order, per-scene JSON loading, text tokenisation, one-hot
category construction, and gzip-pickle IO. Keeping it in one place guarantees
the training cache and the metric cache are built from byte-identical logic.

The two entry points that use this module are:
    * ``inject_to_train_cache.py``  -> writes feature/target .gz files (training)
    * ``inject_to_metric_cache.py`` -> writes per-token gpt.json (evaluation)
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed persona order: Urgency (High/Mid/Low) x Comfort (Low/Mid/High).
# UM_CM is the neutral / default persona. This order is a contract: the one-hot
# category vectors and the per-persona trajectory list are indexed by it, and
# the agent reads them back in exactly this order.
CATEGORIES: List[str] = [
    "UH_CL", "UH_CM", "UH_CH",
    "UM_CL", "UM_CM", "UM_CH",
    "UL_CL", "UL_CM", "UL_CH",
]
NUM_CATS: int = len(CATEGORIES)  # 9

# Defaults shared by both scripts (override on the CLI when needed).
DEFAULT_BERT_BACKBONE: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH: int = 100
# Targets keep the first 8 of the 10 JSON waypoints (DiffusionDrive horizon).
TARGET_TRAJ_LEN: int = 8


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def default_exp_root() -> str:
    """Return ``$NAVSIM_EXP_ROOT`` if set, else an empty string.

    The OpenScene/NavSim setup exports ``NAVSIM_EXP_ROOT`` and stores both the
    training cache and the metric cache under it, so we derive sensible CLI
    defaults from it when available.
    """
    return os.environ.get("NAVSIM_EXP_ROOT", "")


def per_scene_json_path(per_scene_dir: str, token: str) -> str:
    """Return the path of the PersonaDrive annotation JSON for ``token``."""
    return os.path.join(per_scene_dir, f"{token}.json")


# ---------------------------------------------------------------------------
# Per-scene annotation loading
# ---------------------------------------------------------------------------

def load_per_scene(per_scene_dir: str, token: str) -> Dict:
    """Load and validate a single PersonaDrive per-scene annotation.

    :param per_scene_dir: directory holding ``<token>.json`` files
    :param token: OpenScene/NavSim scene token
    :return: the parsed JSON dict (all 9 persona keys guaranteed present)
    :raises FileNotFoundError: if the annotation file does not exist
    :raises KeyError: if any of the 9 persona categories is missing
    """
    path = per_scene_json_path(per_scene_dir, token)
    with open(path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    missing = [c for c in CATEGORIES if c not in scene]
    if missing:
        raise KeyError(f"token {token}: missing persona categories {missing}")
    return scene


def extract_personas(scene: Dict) -> List[str]:
    """Return the 9 ``user_intent`` strings in fixed ``CATEGORIES`` order."""
    return [scene[c]["user_intent"] for c in CATEGORIES]


def extract_trajectories(scene: Dict) -> List[torch.Tensor]:
    """Return the 9 persona trajectories, each a ``[TARGET_TRAJ_LEN, 3]`` tensor.

    The raw JSON stores 10 ``[x, y, heading]`` waypoints per persona; the
    training target uses the first ``TARGET_TRAJ_LEN`` of them.
    """
    trajectories = []
    for c in CATEGORIES:
        traj = scene[c]["trajectory"][:TARGET_TRAJ_LEN]
        trajectories.append(torch.tensor(traj, dtype=torch.float32))  # [8, 3]
    return trajectories


def one_hot_categories() -> List[torch.Tensor]:
    """Return 9 one-hot ``[9]`` tensors, the i-th being one-hot at index i.

    The agent recovers the persona id with ``categories[i].argmax()``, so the
    one-hot must align with the ``CATEGORIES`` order used everywhere else.
    """
    eye = torch.eye(NUM_CATS, dtype=torch.float32)
    return [eye[i] for i in range(NUM_CATS)]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def build_tokenizer(backbone: str = DEFAULT_BERT_BACKBONE):
    """Instantiate a HuggingFace tokenizer for the text backbone.

    Imported lazily so that scripts which never tokenise (and machines without
    ``transformers`` installed) are not forced to pull the dependency in.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(backbone)


def tokenize_personas(
    tokenizer,
    texts: List[str],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Tokenise the 9 persona descriptions into padded tensors for the feature cache.

    :return: ``(input_ids_list, attention_mask_list)`` -- each a list of 9
        tensors of shape ``[max_length]`` (int64), in ``CATEGORIES`` order.
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = [encoded["input_ids"][i] for i in range(NUM_CATS)]
    attention_mask = [encoded["attention_mask"][i] for i in range(NUM_CATS)]
    return input_ids, attention_mask


def tokenize_persona_lists(
    tokenizer,
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, List[int]]:
    """Tokenise one persona description into plain python lists for JSON embedding.

    Matches the format the evaluation fast-path expects when it reads a cached
    ``scene[cat]["tokenized"][backbone]`` entry: a dict with ``input_ids`` and
    ``attention_mask`` as flat lists of ``max_length`` ints.
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ---------------------------------------------------------------------------
# Gzip-pickle IO (NavSim feature/target cache files are gzip-pickled dicts)
# ---------------------------------------------------------------------------

def load_gz(path: str) -> Dict:
    """Load a gzip-pickled dict (a NavSim ``transfuser_*.gz`` cache file)."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def dump_gz(path: str, data: Dict, compresslevel: int = 1) -> None:
    """Write ``data`` back as a gzip-pickled dict (level 1 = fast, like NavSim)."""
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(data, f)
