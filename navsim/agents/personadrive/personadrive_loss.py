from typing import Dict, Optional, Tuple, List
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

from navsim.agents.personadrive.personadrive_config import PersonaDriveConfig
from navsim.agents.personadrive.personadrive_features import BoundingBox2DIndex


def personadrive_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: PersonaDriveConfig
):
    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )
    logits_loss = F.cross_entropy(
        predictions["bert_logits"].view(-1, config.num_purpose_class),
        targets["bio_category"].view(-1).long(),
    )

    # ── Axis-Decomposed Diversity loss ──
    purpose_div_loss = _decomposed_diversity_loss(
        predictions["trajectory"], targets["trajectory"], config
    )

    # ── Hierarchical Guide loss ──
    guide_loss = _hierarchical_guide_loss(predictions, targets)

    loss = (
        config.trajectory_weight * trajectory_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        + config.purpose_div_weight * purpose_div_loss
        + config.logits_class_weight * logits_loss
        + config.guide_weight * guide_loss
    )
    loss_dict = {
        'loss': loss,
        'purpose_div_loss': config.purpose_div_weight * purpose_div_loss,
        'guide_loss': config.guide_weight * guide_loss,
        'trajectory_loss': config.trajectory_weight * trajectory_loss,
        'logits_loss': config.logits_class_weight * logits_loss,
        'agent_class_loss': config.agent_class_weight * agent_class_loss,
        'agent_box_loss': config.agent_box_weight * agent_box_loss,
        'bev_semantic_loss': config.bev_semantic_weight * bev_semantic_loss,
    }
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)
    return loss_dict


# ─────────────────────────────────────────────────────────────
# Hierarchical Guide Loss
# ─────────────────────────────────────────────────────────────

def _traj_length(traj: torch.Tensor) -> torch.Tensor:
    """Total path length for [..., T, 2] trajectories."""
    deltas = traj[..., 1:, :] - traj[..., :-1, :]
    return deltas.norm(dim=-1).sum(dim=-1)  # [...]


def _traj_jerk(traj: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
    """Mean absolute jerk (3rd derivative) for [..., T, 2] trajectories."""
    vel = (traj[..., 1:, :] - traj[..., :-1, :]) / dt      # [..., T-1, 2]
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt         # [..., T-2, 2]
    jrk = (acc[..., 1:, :] - acc[..., :-1, :]) / dt         # [..., T-3, 2]
    return jrk.norm(dim=-1).mean(dim=-1)                     # [...]


def _hierarchical_guide_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    margin_val: float = 0.1,
) -> torch.Tensor:
    """
    Enforce ordering along urgency and comfort axes.

    Persona index mapping (row-major, 3 urgency x 3 comfort):
      0=UH_CL, 1=UH_CM, 2=UH_CH
      3=UM_CL, 4=UM_CM, 5=UM_CH
      6=UL_CL, 7=UL_CM, 8=UL_CH

    Urgency guide: within same comfort column,
      length(UH) > length(UM) > length(UL)  →  high urgency = longer path
    Comfort guide: within same urgency row,
      jerk(CL) > jerk(CM) > jerk(CH)  →  low comfort = more jerk
    """
    pred_xy = predictions["trajectory"][..., :2]   # [B, M, T, 2]
    gt_xy   = targets["trajectory"][..., :2]       # [B, M, T, 2]
    M = pred_xy.shape[1]
    if M != 9:
        return torch.tensor(0.0, device=pred_xy.device)

    loss = torch.tensor(0.0, device=pred_xy.device)
    count = 0

    # ── Urgency guide: length ordering ──
    # columns (same comfort): [0,3,6], [1,4,7], [2,5,8]
    # Within each column: idx 0=UH, 3=UM, 6=UL → length(UH) > length(UM) > length(UL)
    pred_len = _traj_length(pred_xy)   # [B, M]
    gt_len   = _traj_length(gt_xy)     # [B, M]

    for c in range(3):
        uh, um, ul = c, c + 3, c + 6
        pairs = [(uh, um), (um, ul)]  # expected: length[first] > length[second]
        for hi, lo in pairs:
            gt_margin = (gt_len[:, hi] - gt_len[:, lo]).clamp_min(0.0).detach()
            pred_diff = pred_len[:, hi] - pred_len[:, lo]
            loss = loss + F.relu(gt_margin - pred_diff).mean()
            count += 1

    # ── Comfort guide: jerk ordering ──
    # rows (same urgency): [0,1,2], [3,4,5], [6,7,8]
    # Within each row: idx 0=CL, 1=CM, 2=CH → jerk(CL) > jerk(CM) > jerk(CH)
    pred_jerk = _traj_jerk(pred_xy)  # [B, M]
    gt_jerk   = _traj_jerk(gt_xy)    # [B, M]

    for u in range(3):
        cl, cm, ch = u * 3, u * 3 + 1, u * 3 + 2
        pairs = [(cl, cm), (cm, ch)]  # expected: jerk[first] > jerk[second]
        for hi, lo in pairs:
            gt_margin = (gt_jerk[:, hi] - gt_jerk[:, lo]).clamp_min(0.0).detach()
            pred_diff = pred_jerk[:, hi] - pred_jerk[:, lo]
            loss = loss + F.relu(gt_margin - pred_diff).mean()
            count += 1

    return loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Decomposed Diversity Loss
# ─────────────────────────────────────────────────────────────

def _pairwise_kl_margin(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float = 0.25,
    margin_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise-distribution KL alignment for an arbitrary group of M trajectories.
    pred, gt: [B, M, T, C] (uses xy, first 2 dims)
    """
    pred_xy = pred[..., :2]
    gt_xy = gt[..., :2]

    pi, pj = pred_xy.unsqueeze(2), pred_xy.unsqueeze(1)
    D_pred = (pi - pj).abs().mean(dim=(-2, -1))  # [B, M, M]

    gi, gj = gt_xy.unsqueeze(2), gt_xy.unsqueeze(1)
    D_gt = (gi - gj).abs().mean(dim=(-2, -1))    # [B, M, M]

    B, M, _ = D_pred.shape
    diag_mask = torch.eye(M, device=pred.device, dtype=torch.bool).unsqueeze(0)

    D_pred_mask = D_pred.masked_fill(diag_mask, float('-inf'))
    D_gt_mask   = D_gt.masked_fill(diag_mask, float('-inf'))

    P_pred = torch.softmax(D_pred_mask / tau, dim=-1)
    P_gt   = torch.softmax(D_gt_mask / tau, dim=-1)

    eps = 1e-8
    P_pred = P_pred.clamp_min(eps)
    P_gt   = P_gt.clamp_min(eps)

    log_P_pred = P_pred.log()
    log_P_gt   = P_gt.log()

    kl_gt_pred = F.kl_div(log_P_gt, P_pred, reduction='none').sum(dim=-1)
    kl_pred_gt = F.kl_div(log_P_pred, P_gt, reduction='none').sum(dim=-1)
    kl_loss = 0.5 * (kl_gt_pred + kl_pred_gt).mean()

    triu_mask = torch.triu(torch.ones(M, M, device=pred.device, dtype=torch.bool), diagonal=1)
    pair_dists_pred = D_pred[:, triu_mask]
    margin = D_gt[:, triu_mask]
    margin_loss = F.relu(margin - pair_dists_pred).mean()

    return kl_loss + margin_alpha * margin_loss


def _decomposed_diversity_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: PersonaDriveConfig,
) -> torch.Tensor:
    """
    Within-axis + weighted cross-axis diversity loss.

    urgency_groups (same comfort, diff urgency): [[0,3,6], [1,4,7], [2,5,8]]
    comfort_groups (same urgency, diff comfort): [[0,1,2], [3,4,5], [6,7,8]]
    """
    M = predictions.shape[1]
    if M != 9:
        return _purpose_diversity_KL_loss(
            predictions, targets,
            tau=config.diversity_tau, margin_alpha=config.diversity_margin_alpha,
        )

    tau = config.diversity_tau
    margin_alpha = config.diversity_margin_alpha
    w_cross = config.diversity_cross_weight

    urgency_groups = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    comfort_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    all_groups = urgency_groups + comfort_groups

    within_loss = torch.tensor(0.0, device=predictions.device)
    for grp in all_groups:
        within_loss = within_loss + _pairwise_kl_margin(
            predictions[:, grp], targets[:, grp], tau=tau, margin_alpha=margin_alpha,
        )
    within_loss = within_loss / len(all_groups)

    cross_loss = _pairwise_kl_margin(
        predictions, targets, tau=tau, margin_alpha=margin_alpha,
    )

    return within_loss + w_cross * cross_loss


# ─────────────────────────────────────────────────────────────
# Original Diversity Loss (kept for backward compat / fallback)
# ─────────────────────────────────────────────────────────────

def _purpose_diversity_KL_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tau: float = 0.25,
    margin_alpha: float = 1.0,
) -> torch.Tensor:
    pred = predictions[..., :2]
    gt = targets[..., :2]

    pi, pj = pred.unsqueeze(2), pred.unsqueeze(1)
    D_pred = (pi - pj).abs().mean(dim=(-2, -1))

    gi, gj = gt.unsqueeze(2), gt.unsqueeze(1)
    D_gt = (gi - gj).abs().mean(dim=(-2, -1))

    B, M, _ = D_pred.shape
    diag_mask = torch.eye(M, device=pred.device, dtype=torch.bool).unsqueeze(0)

    D_pred_mask = D_pred.masked_fill(diag_mask, float('-inf'))
    D_gt_mask   = D_gt.masked_fill(diag_mask, float('-inf'))

    P_pred = torch.softmax(D_pred_mask / tau, dim=-1)
    P_gt   = torch.softmax(D_gt_mask / tau, dim=-1)

    eps = 1e-8
    P_pred = P_pred.clamp_min(eps)
    P_gt   = P_gt.clamp_min(eps)

    log_P_pred = P_pred.log()
    log_P_gt   = P_gt.log()

    kl_gt_pred = F.kl_div(log_P_gt,   P_pred, reduction='none').sum(dim=-1)
    kl_pred_gt = F.kl_div(log_P_pred, P_gt,   reduction='none').sum(dim=-1)
    loss = 0.5 * (kl_gt_pred + kl_pred_gt).mean()

    triu_mask = torch.triu(torch.ones(M, M, device=pred.device, dtype=torch.bool), diagonal=1)
    pair_dists_pred = D_pred[:, triu_mask]
    margin = D_gt[:, triu_mask]
    margin_loss = F.relu(margin - pair_dists_pred).mean()
    loss = loss + margin_alpha * margin_loss

    return loss


# ─────────────────────────────────────────────────────────────
# Agent Loss (Hungarian matching)
# ─────────────────────────────────────────────────────────────

def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: PersonaDriveConfig
):
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )
        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)
    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
