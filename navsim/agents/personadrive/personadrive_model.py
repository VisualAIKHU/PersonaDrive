from typing import Dict
import io
import math
import copy
from torchvision.transforms.functional import to_pil_image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.personadrive.personadrive_config import PersonaDriveConfig
from navsim.agents.personadrive.personadrive_backbone import PersonaDriveBackbone
from navsim.agents.personadrive.personadrive_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.personadrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
from navsim.agents.personadrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, \
                                                        GridSampleCrossBEVAttention, PCMF_Fuser, BERT
from navsim.agents.personadrive.modules.multimodal_loss import LossComputer


class PersonaDriveModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: PersonaDriveConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [1, config.num_bounding_boxes, config.num_purpose_class]

        self._config = config
        self._backbone = PersonaDriveBackbone(config)

        self._bert = BERT(backbone=config.bert_backbone, num_classes=config.num_purpose_class)

        self._keyval_embedding = nn.Embedding(8**2 + 1 + config.num_purpose_class, config.tf_d_model)  # 8x8 feature grid + trajectory + user input
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        self._user_encoding = nn.Linear(256, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        
        self.bert_proj = nn.Sequential(
            nn.LayerNorm(config.bert_hidden_size),
            nn.Linear(config.bert_hidden_size, config.tf_d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(config.tf_d_model),
        )
        self.bert_clf = nn.Sequential(
            nn.LayerNorm(config.bert_hidden_size),
            nn.Linear(config.bert_hidden_size, config.tf_d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.tf_d_model, config.num_purpose_class),
        )

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        
        camera_feature:  torch.Tensor = features["camera_feature"]   # [bs, 3, 256, 1024]
        lidar_feature:   torch.Tensor = features["lidar_feature"]    # [bs, 1, 256, 256]
        status_feature:  torch.Tensor = features["status_feature"]   # [bs, 8]
        bert_feature:    torch.Tensor = features["bert_feature"]     # [bs, M, 100, 384]
        attention_mask:  torch.Tensor = features["attention_mask"]   # [bs, M, 100]

        batch_size, M, _ = attention_mask.shape # bs, 1 or 3, 100 

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature) # [bs, 64, 64, 64], [bs, 512, 8, 8]
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1) # [bs, 64, 256]
        bev_feature = bev_feature.permute(0, 2, 1) # [bs, 256, 64]

        status_encoding = self._status_encoding(status_feature) # [bs, 256]

        flat_ids = bert_feature.reshape(batch_size * M, -1) # [bs * M, 100]
        flat_mask = attention_mask.reshape(batch_size * M, -1) # [bs * M, 100]
        user_text_emb, user_text_logits = self._bert(flat_ids, flat_mask) # [bs, M, 256]
        user_text_emb, user_text_logits = user_text_emb.view(batch_size, M, 256), user_text_logits.view(batch_size, M, self._config.num_purpose_class)
        user_encoding = self._user_encoding(user_text_emb) # [bs, 1, 256]


        keyval = torch.concatenate([bev_feature, status_encoding[:, None], user_encoding], dim=1) # [bs, 65+M, 256]
        W = self._keyval_embedding.weight
        if M == self._config.num_purpose_class:
            keyval += W.unsqueeze(0) # [bs, 65+M, 256]
        else:
            keyval[:, :65, :] += W[:65].unsqueeze(0) # [bs, 65, 256]
            mode_idx = user_text_logits.argmax(dim=-1).squeeze(1).long() # [bs,] 
            keyval[:, 65, :] += W[65 + mode_idx]
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1) # [bs, 33, 256]
        query_out = self._tf_decoder(query, keyval) # [bs, 33, 256]

        concat_cross_bev = keyval[:, :-1-M].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]) # [bs, 256, 8, 8]
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False) # [bs, 256, 64, 64]
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1) # [bs, 320, 64, 64]

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1)) # [bs, 4096, 256]
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]) # [bs, 256, 64, 64]

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale) # [bs, 7, 128, 256]
        trajectory_query, agents_query, user_query = query_out.split(self._query_splits, dim=1) # [bs, 1, 256], [bs, 30, 256], [bs, 1, 256]
        if M == 1:
            b = torch.arange(batch_size, device=user_query.device)
            mode_idx = user_text_logits.argmax(dim=-1).squeeze(1).long() # [bs,] 
            user_query = user_query[b, mode_idx, :].unsqueeze(1) # [bs, 1, 256]

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map, "bert_logits": user_text_logits} 
        trajectory = self._trajectory_head(trajectory_query, agents_query, user_query,
                                           cross_bev_feature, bev_spatial_shape, status_encoding[:, None],
                                           user_text_logits=user_text_logits, 
                                           targets=targets, global_img=None)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class ExpertMemory(nn.Module):
    def __init__(self, memory_size: int, input_dim: int):
        super().__init__()
        self.key_memory = nn.Parameter(torch.randn(memory_size, input_dim))
        self.value_memory = nn.Parameter(torch.randn(memory_size, input_dim))

    def reset_parameters(self):
        nn.init.orthogonal_(self.key_memory)
        nn.init.orthogonal_(self.value_memory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x_norm = F.normalize(x, dim=-1)
        key_memory_norm = F.normalize(self.key_memory, dim=-1)  # [M, D]
        value = self.value_memory  # [M, D]

        sim = torch.matmul(x_norm, key_memory_norm.T)  # [B, N, M]
        addressing = F.softmax(sim / 0.1, dim=-1)  # [B, N, M]
        attended = torch.matmul(addressing, value)  # [B, N, D]
        return attended

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
    
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )

        specs = {}
        specs["agent_"] = (config.tf_d_model, config.slots_per_source.get("agent", 16))
        specs["ego"]   = (config.tf_d_model, config.slots_per_source.get("ego", 16))
        specs["user"]  = (config.tf_d_model, config.slots_per_source.get("user", 16))

        self.pcmf = PCMF_Fuser(
            specs=specs,
            d_model=config.tf_d_model,
            num_heads=config.tf_num_head,
            dropout=config.tf_dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm_ffn = nn.LayerNorm(config.tf_d_model)

        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                user_query,
                time_embed, 
                status_encoding,
                global_img=None):

        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)

        sources = {
            "agent_": agents_query,   # [B,La,d]
            "ego":   ego_query,      # [B,Le,d]
            "user":  user_query,     # [B,Lu,d]
        }

        traj_feature = self.pcmf(
            traj_feature=traj_feature,
            sources=sources,
            time_embed=time_embed,
            status_encoding=status_encoding,
        )  # [B,Q,d]

        traj_feature = self.norm_ffn(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)

        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * math.pi
        return poses_reg, poses_cls

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                user_query,
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, 
                                       agents_query, ego_query, user_query,
                                       time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: PersonaDriveConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # [20, 8, 2] -> 20 modes x 8 poses x (x,y)

        # ── Hierarchical conditioning (urgency + comfort) ──
        self.urgency_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128))
        self.comfort_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128))
        self.alpha_urgency = nn.Linear(128, 1)                                            # scalar
        self.alpha_comfort = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 8))  # per-timestep

        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        self._config = config
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def anchor_(self):
        pass

    def _compute_hierarchical_anchor(self, gate_in, base_anchor, m):
        """Compute persona-conditioned anchor via urgency (global) + comfort (per-timestep).
        Args:
            gate_in: [bs, 256] - user query feature for this persona
            base_anchor: [bs, 20, 8, 2]
            m: persona index (0-8)
        Returns:
            plan_anchor_m: [bs, 20, 8, 2] - conditioned anchor
        """
        neutral = self._config.neutral_persona_idx  # default 4 = UM_CM
        if m == neutral:
            return base_anchor

        urgency_feat = self.urgency_proj(gate_in)     # [bs, 128]
        comfort_feat = self.comfort_proj(gate_in)     # [bs, 128]

        alpha_u = self.alpha_urgency(urgency_feat).view(-1, 1, 1, 1)         # global scale
        plan_anchor_u = base_anchor * alpha_u                                 # [bs, 20, 8, 2]

        alpha_c = (self.alpha_comfort(comfort_feat).sigmoid()
                   * self._config.delta_c + self._config.gamma_c)            # range [gamma_c, gamma_c+delta_c], shape [bs, 8]
        plan_anchor_m = plan_anchor_u * alpha_c.view(-1, 1, 8, 1)            # per-timestep modulation

        return plan_anchor_m

    def forward(self, ego_query, agents_query, user_query, bev_feature, bev_spatial_shape, status_encoding, user_text_logits=None, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, user_query, bev_feature, bev_spatial_shape, status_encoding, user_text_logits, targets, global_img)
        else:
            return self.forward_test(ego_query, agents_query, user_query, bev_feature, bev_spatial_shape, status_encoding, user_text_logits, global_img)

    def forward_train(self, ego_query, agents_query, user_query,
                            bev_feature, bev_spatial_shape, status_encoding, 
                            user_text_logits, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        M = user_query.shape[1]

        base_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # [bs, 20, 8, 2]

        trajectory_loss_dict = {}
        purpose_wised_loss = {}
        ret_traj_loss = 0.0

        per_mode_best = []
        basis = []

        for m in range(M):
            uq = user_query[:, m, :].unsqueeze(1)  # [bs, 1, 256]

            gate_in = uq.squeeze(1)  # [bs, 256]
            plan_anchor_m = self._compute_hierarchical_anchor(gate_in, base_anchor, m)
            neutral = self._config.neutral_persona_idx
            basis.append(base_anchor if m == neutral else plan_anchor_m)

            odo_info_fut = self.norm_odo(plan_anchor_m)
            timesteps = torch.randint(0, 50, (bs,), device=device)
            noise = torch.randn(odo_info_fut.shape, device=device)

            noisy = self.diffusion_scheduler.add_noise(
                original_samples=odo_info_fut, noise=noise, timesteps=timesteps
            ).float().clamp_(-1, 1)
            noisy_traj_points = self.denorm_odo(noisy)

            ego_fut_mode = noisy_traj_points.shape[1]

            # proj noisy_traj_points -> query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)

            # time embed
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, uq, time_embed, status_encoding, global_img
            )

            tgt_m = targets['trajectory'][:, m, :]

            for poses_reg, poses_cls in zip(poses_reg_list, poses_cls_list):
                neutral = self._config.neutral_persona_idx
                loss_m = self.loss_computer(poses_reg, poses_cls, tgt_m, base_anchor if m == neutral else plan_anchor_m)
                purpose_wised_loss[m] = purpose_wised_loss.get(m, 0.0) + loss_m

            mode_idx = poses_cls_list[-1].argmax(dim=-1)
            mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
            per_mode_best.append(torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1))

        for idx in sorted(purpose_wised_loss.keys()):
            loss_m = purpose_wised_loss[idx] 
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = loss_m
            ret_traj_loss = ret_traj_loss + loss_m / float(M)

        best_reg = torch.stack(per_mode_best, dim=1)  # [bs, M, num_poses, 3]
        return {"trajectory": best_reg, "trajectory_loss": ret_traj_loss, "trajectory_loss_dict": trajectory_loss_dict, "basis": basis}
        
    def forward_test(self, ego_query,agents_query, user_query,
                    bev_feature,bev_spatial_shape,status_encoding,user_text_logits,global_img) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        base_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # [bs, 20, 8, 2]
        M = user_query.shape[1] if user_query.dim() == 3 else 1
        per_mode_best = []
        for m in range(M):
            if M == 1:
                uq = user_query  # [bs, 1, 256]
            else:
                uq = user_query[:, m, :].unsqueeze(1)  # [bs, 1, 256]

            if M == 1:
                logits_m = user_text_logits[:, 0, :]
                persona_idx = torch.softmax(logits_m, dim=-1)
                neutral = self._config.neutral_persona_idx
                if persona_idx.argmax(dim=-1).item() == neutral:
                    plan_anchor_m = base_anchor
                else:
                    gate_in = uq.squeeze(1)
                    plan_anchor_m = self._compute_hierarchical_anchor(gate_in, base_anchor, persona_idx.argmax(dim=-1).item())
            else:
                gate_in = uq.squeeze(1)
                plan_anchor_m = self._compute_hierarchical_anchor(gate_in, base_anchor, m)

            img = self.norm_odo(plan_anchor_m)
            import os as _os
            _seed_str = _os.environ.get("PD_INFER_SEED")
            if _seed_str is not None:
                _gen = torch.Generator(device=device).manual_seed(int(_seed_str) * 100003 + m * 7919)
                noise = torch.randn(img.shape, device=device, generator=_gen)
            else:
                noise = torch.randn(img.shape, device=device)
            trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
            img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
            ego_fut_mode = img.shape[1]
            for k in roll_timesteps[:]:
                x_boxes = torch.clamp(img, min=-1, max=1)
                noisy_traj_points = self.denorm_odo(x_boxes)

                traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

                timesteps = k
                if not torch.is_tensor(timesteps):
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(img.device)
                timesteps = timesteps.expand(img.shape[0])
                time_embed = self.time_mlp(timesteps)
                time_embed = time_embed.view(bs,1,-1)

                poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, 
                                                                agents_query, ego_query, uq, time_embed, status_encoding,global_img)
                poses_reg = poses_reg_list[-1]
                poses_cls = poses_cls_list[-1]
                x_start = poses_reg[...,:2]
                x_start = self.norm_odo(x_start)
                img = self.diffusion_scheduler.step(
                    model_output=x_start,
                    timestep=k,
                    sample=img
                ).prev_sample

            mode_idx = poses_cls.argmax(dim=-1)
            mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
            best_reg_m = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
            per_mode_best.append(best_reg_m)

        best_reg = torch.stack(per_mode_best, dim=1)
        return {"trajectory": best_reg}
