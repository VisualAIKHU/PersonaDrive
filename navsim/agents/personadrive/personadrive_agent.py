from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.personadrive.personadrive_config import PersonaDriveConfig

from navsim.agents.personadrive.personadrive_model import PersonaDriveModel

from navsim.agents.personadrive.personadrive_callback import PersonaDriveCallback 
from navsim.agents.personadrive.personadrive_loss import personadrive_loss
from navsim.agents.personadrive.personadrive_features import PersonaDriveFeatureBuilder, PersonaDriveTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.personadrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from pytorch_lightning.callbacks import ModelCheckpoint

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class PersonaDriveAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: PersonaDriveConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._model = PersonaDriveModel(config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}

            # Backward compatibility with checkpoints trained before the rename:
            #   model attribute  '_transfuser_model.' -> '_model.'
            #   fusion module     '.qccm.'            -> '.pcmf.'
            state_dict = {k.replace('_transfuser_model.', '_model.'): v for k, v in state_dict.items()}
            state_dict = {k.replace('.qccm.', '.pcmf.'): v for k, v in state_dict.items()}

            # Filter out keys with shape mismatch
            model_state = self.state_dict()
            skipped_keys = []
            for k in list(state_dict.keys()):
                if k in model_state and state_dict[k].shape != model_state[k].shape:
                    skipped_keys.append(k)
                    del state_dict[k]
            if skipped_keys:
                print(f"Skipped keys due to shape mismatch: {skipped_keys}")

            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        # Backward compatibility for pre-rename checkpoints.
        state_dict = {k.replace("_transfuser_model.", "_model."): v for k, v in state_dict.items()}
        state_dict = {k.replace(".qccm.", ".pcmf."): v for k, v in state_dict.items()}
        model_state = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if k not in model_state or v.shape == model_state[k].shape}
        self.load_state_dict(state_dict, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass.

        PersonaDriveTargetBuilder loads ``transfuser_target.gz``, which holds the
        agent/BEV targets plus the per-persona ``trajectories`` (list of 9) and
        ``categories`` injected into it. ``transform_batch`` below consumes
        ``targets['trajectories']`` for multi-persona diffusion training.
        """
        return [
            PersonaDriveTargetBuilder(config=self._config),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [PersonaDriveFeatureBuilder(config=self._config)]

    def transform_batch(self, batch):
        """Override: stack all 9 modes for multi-mode diffusion training."""
        features, targets = batch
        _features = {
            'camera_feature': features['camera_feature'],
            'lidar_feature': features['lidar_feature'],
            'status_feature': features['status_feature'],
            'bert_feature': torch.stack([features['input_ids'][i] for i in range(9)], dim=1),      # [bs, 9, 100]
            'attention_mask': torch.stack([features['attention_mask'][i] for i in range(9)], dim=1),  # [bs, 9, 100]
        }
        _targets = {
            'agent_states': targets['agent_states'],
            'agent_labels': targets['agent_labels'],
            'bev_semantic_map': targets['bev_semantic_map'],
            'trajectory': torch.stack([targets['trajectories'][i] for i in range(9)], dim=1),  # [bs, 9, 8, 3]
            'bio_category': torch.stack([targets['categories'][i].argmax(dim=1) for i in range(9)], dim=1),  # [bs, 9]
        }
        return _features, _targets

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._model(features, targets=targets)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return personadrive_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        # return [PersonaDriveCallback(self._config)]
        checkpoint_cb = ModelCheckpoint(
            filename="epoch{epoch:03d}-valloss{val/loss:.3f}",
            monitor="val/loss",
            mode="min",
            save_top_k=-1,          # save every epoch
            every_n_epochs=1,
            save_last=True,
            save_on_train_epoch_end=True,  # save at the end of each training epoch
        )
        return [PersonaDriveCallback(self._config), checkpoint_cb]
