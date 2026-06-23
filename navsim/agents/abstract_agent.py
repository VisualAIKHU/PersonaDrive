from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl
import time

from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
import numpy as np

class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    def transform_batch(self, batch):
        """
        Transform raw cache batch into agent-specific features/targets.
        Default: randomly select 1 of 9 persona modes.
        Override in subclass for multi-mode agents.
        """
        import random
        features, targets = batch
        mode_idx = random.randint(0, 8)
        _features = {
            'camera_feature': features['camera_feature'],
            'lidar_feature': features['lidar_feature'],
            'status_feature': features['status_feature'],
            'bert_feature': features['input_ids'][mode_idx],       # [bs, 100]
            'attention_mask': features['attention_mask'][mode_idx], # [bs, 100]
        }
        _targets = {
            'agent_states': targets['agent_states'],
            'agent_labels': targets['agent_labels'],
            'bev_semantic_map': targets['bev_semantic_map'],
            'trajectory': targets['trajectories'][mode_idx].float(), # [bs, 8, 3]
        }
        return _features, _targets

    def forward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :param targets: Optional dictionary of targets (used by some agents during training).
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            t = time.time()
            predictions = self.forward(features)
            # poses = predictions["trajectory"].squeeze(dim=0).numpy()
            poses = predictions["trajectory"].squeeze(dim=(0, 1)).numpy()

        # extract trajectory
        return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []
