from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import json
import numpy as np
import time

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import Trajectory
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from concurrent.futures import ProcessPoolExecutor, as_completed

from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax, configure_ax

# The 9 persona categories (fixed order)
CATEGORIES = [
    "UH_CL", "UH_CM", "UH_CH",
    "UM_CL", "UM_CM", "UM_CH",
    "UL_CL", "UL_CM", "UL_CH",
]

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


# ==================== Accuracy (ADE/FDE) helpers ====================
def _ade_fde(pred_xy, gt_xy):
    """pred_xy: (T,2), gt_xy: (T,2) -> (ADE, FDE)"""
    T = min(pred_xy.shape[0], gt_xy.shape[0])
    diffs = np.linalg.norm(pred_xy[:T] - gt_xy[:T], axis=-1)
    return float(diffs.mean()), float(diffs[-1])


def _compute_token_metrics(token, pred_traj_root, per_scene_dir, cats):
    """Compute ADE/FDE for one token."""
    row = {"token": token}
    pred_trajs = {}
    for cat in cats:
        npy_path = os.path.join(pred_traj_root, cat, f"{token}.npy")
        if not os.path.exists(npy_path):
            return None
        pred_trajs[cat] = np.load(npy_path).astype(np.float32)

    # ADE/FDE per category
    json_path = os.path.join(per_scene_dir, f"{token}.json")
    if not os.path.exists(json_path):
        return row
    with open(json_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    for cat in cats:
        if cat not in scene_data or "trajectory" not in scene_data[cat]:
            continue
        gt_xy = np.array(scene_data[cat]["trajectory"], dtype=np.float32)[:, :2]
        pred_xy = pred_trajs[cat][:, :2]
        ade, fde = _ade_fde(pred_xy, gt_xy)
        row[f"ADE_{cat}"] = ade
        row[f"FDE_{cat}"] = fde

    return row


# ==================== Visualization helpers ====================
CAT_NAMES = [
    "Urg.H / Comf.L", "Urg.H / Comf.M", "Urg.H / Comf.H",
    "Urg.M / Comf.L", "Urg.M / Comf.M", "Urg.M / Comf.H",
    "Urg.L / Comf.L", "Urg.L / Comf.M", "Urg.L / Comf.H",
]
CAT_COLORS = {
    "UH_CL": "#E85050", "UH_CM": "#ED7373", "UH_CH": "#F19696",
    "UM_CL": "#5090D0", "UM_CM": "#73A6D9", "UM_CH": "#96BCE3",
    "UL_CL": "#50B870", "UL_CM": "#73C68D", "UL_CH": "#96D4A9",
}
KEY_TO_NAME = dict(zip(CATEGORIES, CAT_NAMES))


def _total_distance(trajectory):
    poses = trajectory.poses[:, :2]
    diffs = np.diff(poses, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _has_backward(trajectory):
    return bool(np.any(trajectory.poses[:, 0] < 0))


def _check_distance_order(pred_trajs):
    dists = {k: _total_distance(pred_trajs[k]) for k in CATEGORIES}
    for u in ["UH", "UM", "UL"]:
        if not (dists[f"{u}_CL"] > dists[f"{u}_CM"] > dists[f"{u}_CH"]):
            return False, dists
    for c in ["CL", "CM", "CH"]:
        if not (dists[f"UH_{c}"] > dists[f"UM_{c}"] > dists[f"UL_{c}"]):
            return False, dists
    return True, dists


def _save_vis_grid(scene, pred_trajs, dists, save_path):
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, key in enumerate(CATEGORIES):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
        color = CAT_COLORS[key]
        traj_config = {
            "fill_color": color, "fill_color_alpha": 1.0,
            "line_color": color, "line_color_alpha": 1.0,
            "line_width": 2.5, "line_style": "-",
            "marker": "o", "marker_size": 5,
            "marker_edge_color": "black", "zorder": 3,
        }
        add_trajectory_to_bev_ax(ax, pred_trajs[key], traj_config)
        configure_bev_ax(ax)
        configure_ax(ax)
        ax.set_title(f"{KEY_TO_NAME[key]} (d={dists[key]:.1f}m)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # one-hot vs text branch
    use_onehot = OmegaConf.select(cfg, "agent.config.use_onehot", default=False)

    # resolve bert_backbone name + fallback tokenizer
    bert_backbone = OmegaConf.select(cfg, "agent.config.bert_backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = None  # lazy load (only when no pre-tokenized data)

    # validate driving_purpose
    driving_purpose = cfg.driving_purpose
    use_all = (driving_purpose == "all")
    if not use_all and driving_purpose not in CATEGORIES:
        raise ValueError(f"driving_purpose must be one of {CATEGORIES} or 'all', got '{driving_purpose}'")

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # list of categories to evaluate
    cats_to_eval = CATEGORIES if use_all else [driving_purpose]

    # Visualization config
    vis_enabled = OmegaConf.select(cfg, "vis", default=False) and use_all
    if vis_enabled:
        vis_dir = os.path.join(OmegaConf.select(cfg, "vis_dir", default="vis_results"), cfg.experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
    vis_saved = 0

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_evaluate):
        # t = time.time()
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # load user_intent from the per_scene JSON
            per_scene_path = os.path.join(cfg.per_scene_dir, f"{token}.json")
            with open(per_scene_path, "r", encoding="utf-8") as jf:
                scene_data = json.load(jf)

            agent_input = scene_loader.get_agent_input_from_token(token)

            # per-category M=1 inference + PDM score
            pred_trajectories = {}
            for cat in cats_to_eval:
                score_row: Dict[str, Any] = {"token": token, "category": cat, "valid": True}
                try:
                    if agent.requires_scene:
                        scene = scene_loader.get_scene_from_token(token)
                        trajectory = agent.compute_trajectory(agent_input, scene)
                    else:
                        if use_onehot:
                            # one-hot: category index -> [9] one-hot vector
                            cat_idx = CATEGORIES.index(cat)
                            onehot = [0.0] * 9
                            onehot[cat_idx] = 1.0
                            agent_input.bert_feature = [onehot]       # [[9 floats]] → tensor [1, 9]
                            agent_input.attention_mask = [[1.0] * 9]  # dummy [1, 9]
                        else:
                            # prefer pre-tokenized data; otherwise tokenize on the fly
                            pre_tok = scene_data[cat].get("tokenized", {}).get(bert_backbone)
                            if pre_tok is not None:
                                encoded = pre_tok
                            else:
                                if tokenizer is None:
                                    tokenizer = AutoTokenizer.from_pretrained(bert_backbone)
                                encoded = tokenizer(
                                    scene_data[cat]["user_intent"],
                                    padding="max_length", truncation=True,
                                    max_length=100, return_tensors=None,
                                )
                            agent_input.bert_feature = [encoded["input_ids"]]           # [[100 ints]] → tensor [1, 100]
                            agent_input.attention_mask = [encoded["attention_mask"]]     # [[100 ints]] → tensor [1, 100]
                        trajectory = agent.compute_trajectory(agent_input)

                    pred_trajectories[cat] = trajectory
                    os.makedirs(f"Diversity/{cfg.experiment_name}/{cat}", exist_ok=True)
                    np.save(f"Diversity/{cfg.experiment_name}/{cat}/{token}.npy", trajectory.poses)

                    pdm_result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=trajectory,
                        future_sampling=simulator.proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    score_row.update(asdict(pdm_result))
                except Exception as e:
                    logger.warning(f"----------- Agent failed for token {token}, category {cat}:")
                    traceback.print_exc()
                    score_row["valid"] = False
                pdm_results.append(score_row)

            # Visualization: check conditions and save 3x3 grid
            if vis_enabled and len(pred_trajectories) == 9:
                if not any(_has_backward(pred_trajectories[k]) for k in CATEGORIES):
                    order_ok, dists = _check_distance_order(pred_trajectories)
                    if order_ok:
                        scene = scene_loader.get_scene_from_token(token)
                        _save_vis_grid(scene, pred_trajectories, dists,
                                       os.path.join(vis_dir, f"{token}.png"))
                        vis_saved += 1

        except Exception as e:
            logger.warning(f"----------- Failed to load data for token {token}:")
            traceback.print_exc()
            for cat in cats_to_eval:
                pdm_results.append({"token": token, "category": cat, "valid": False})

        # print(f"{time.time() - t:.2f}s")
    if vis_enabled:
        logger.info(f"Visualization: saved {vis_saved} images to {vis_dir}")
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    use_all = (cfg.driving_purpose == "all")
    cats_to_eval = CATEGORIES if use_all else [cfg.driving_purpose]
    for cat in cats_to_eval:
        os.makedirs(f"Diversity/{cfg.experiment_name}/{cat}", exist_ok=True)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")

    # Limit number of scenes if max_scenes is set
    max_scenes = OmegaConf.select(cfg, "max_scenes", default=-1)
    if max_scenes > 0 and len(tokens_to_evaluate) > max_scenes:
        import random
        random.seed(42)
        tokens_to_evaluate = sorted(random.sample(tokens_to_evaluate, max_scenes))
        logger.info(f"Limiting evaluation to {max_scenes} scenes (seed=42)")
    selected_tokens = set(tokens_to_evaluate)

    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [t for t in tokens_list if t in selected_tokens],
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    data_points = [dp for dp in data_points if dp["tokens"]]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Reported metric: PDMS (the PDM sub-scores are aggregated into it internally).
    _pdm_cols = [
        ("PDMS", "score"),
    ]

    def _safe_mean(df, col):
        if col in df.columns and len(df) > 0:
            return df[col].mean()
        return float("nan")

    # print per-category results
    header_metrics = " | ".join(f"{name:>7s}" for name, _ in _pdm_cols)
    sep_width = 12 + len(header_metrics) + 14
    summary_lines = [f"\n{'='*sep_width}", f"  Driving purpose: {cfg.driving_purpose}", f"{'='*sep_width}"]
    summary_lines.append(f"  {'Category':8s} | {header_metrics} | {'valid':>10s}")
    summary_lines.append(f"{'─'*sep_width}")
    for cat in cats_to_eval:
        cat_df = pdm_score_df[pdm_score_df["category"] == cat]
        n_valid = cat_df["valid"].sum()
        n_total = len(cat_df)
        cat_valid = cat_df[cat_df["valid"]]
        vals = " | ".join(f"{_safe_mean(cat_valid, col):7.4f}" for _, col in _pdm_cols)
        summary_lines.append(f"  {cat:8s} | {vals} | {n_valid}/{n_total}")

    # overall average
    all_valid = pdm_score_df[pdm_score_df["valid"]]
    overall_vals = " | ".join(f"{_safe_mean(all_valid, col):7.4f}" for _, col in _pdm_cols)
    overall_avg = _safe_mean(all_valid, "score")
    summary_lines.append(f"{'─'*sep_width}")
    summary_lines.append(f"  {'AVERAGE':8s} | {overall_vals} | {len(all_valid)}/{len(pdm_score_df)}")
    summary_lines.append(f"{'='*sep_width}")

    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)
    print(summary_text)

    # save CSV
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    # also save a summary CSV with per-category average rows
    summary_rows = []
    for cat in cats_to_eval:
        cat_df = pdm_score_df[pdm_score_df["category"] == cat]
        cat_valid = cat_df[cat_df["valid"]]
        if len(cat_valid) > 0:
            avg_row = cat_valid.drop(columns=["token", "category", "valid"]).mean(skipna=True).to_dict()
            avg_row.update({"token": "average", "category": cat, "valid": True})
            summary_rows.append(avg_row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(save_path / f"{timestamp}_summary.csv", index=False)

    # ==================== Diversity + ADE/FDE ====================
    diversity_root = f"Diversity/{cfg.experiment_name}"
    per_scene_dir = cfg.per_scene_dir

    # Find tokens with npy files in all evaluated categories
    token_sets = []
    for cat in cats_to_eval:
        cat_dir = os.path.join(diversity_root, cat)
        if os.path.isdir(cat_dir):
            token_sets.append({f.replace(".npy", "") for f in os.listdir(cat_dir) if f.endswith(".npy")})
    if token_sets:
        common_tokens = sorted(set.intersection(*token_sets))
        logger.info(f"Computing ADE/FDE for {len(common_tokens)} tokens...")

        metric_rows = []
        with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as ex:
            futures = {
                ex.submit(_compute_token_metrics, t, diversity_root, per_scene_dir, cats_to_eval): t
                for t in common_tokens
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    metric_rows.append(result)

        if metric_rows:
            metric_lines = [f"\n{'='*60}", "  Accuracy (ADE / FDE)", f"{'='*60}"]
            metric_lines.append(f"  {'Category':8s} | {'ADE':>8s} | {'FDE':>8s}")
            metric_lines.append(f"{'─'*60}")

            cat_ade_fde = {}
            for cat in cats_to_eval:
                ade_vals = [r[f"ADE_{cat}"] for r in metric_rows if isinstance(r.get(f"ADE_{cat}"), float)]
                fde_vals = [r[f"FDE_{cat}"] for r in metric_rows if isinstance(r.get(f"FDE_{cat}"), float)]
                if ade_vals:
                    mean_ade, mean_fde = np.mean(ade_vals), np.mean(fde_vals)
                    cat_ade_fde[cat] = (mean_ade, mean_fde)
                    metric_lines.append(f"  {cat:8s} | {mean_ade:8.4f} | {mean_fde:8.4f}")
                else:
                    metric_lines.append(f"  {cat:8s} | {'N/A':>8s} | {'N/A':>8s}")

            if cat_ade_fde:
                avg_ade = np.mean([v[0] for v in cat_ade_fde.values()])
                avg_fde = np.mean([v[1] for v in cat_ade_fde.values()])
                metric_lines.append(f"{'─'*60}")
                metric_lines.append(f"  {'AVERAGE':8s} | {avg_ade:8.4f} | {avg_fde:8.4f}")
            metric_lines.append(f"{'='*60}")

            metric_text = "\n".join(metric_lines)
            logger.info(metric_text)
            print(metric_text)

            # Append to summary CSV
            if summary_rows:
                for row in summary_rows:
                    cat = row["category"]
                    if cat in cat_ade_fde:
                        row["ADE"] = cat_ade_fde[cat][0]
                        row["FDE"] = cat_ade_fde[cat][1]
                # Add overall average row
                summary_rows.append({
                    "token": "average", "category": "ALL", "valid": True,
                    "score": overall_avg,
                    "ADE": avg_ade if cat_ade_fde else float("nan"),
                    "FDE": avg_fde if cat_ade_fde else float("nan"),
                })
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(save_path / f"{timestamp}_summary.csv", index=False)

    logger.info(f"Results stored in: {save_path / f'{timestamp}.csv'}")


if __name__ == "__main__":
    main()
