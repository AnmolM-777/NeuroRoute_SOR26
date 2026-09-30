import os
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.ai.models import DQN, DuelingDQN

logger = logging.getLogger(__name__)


class PretrainedModelRegistry:
    """
    Model registry for storing, versioning, and loading pre-trained routing neural networks.
    """
    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def save_model(self, model: torch.nn.Module, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        filepath = os.path.join(self.registry_dir, f"{name}.pt")
        save_dict = {
            "model_state": model.state_dict(),
            "model_type": model.__class__.__name__,
            "metadata": metadata or {}
        }
        torch.save(save_dict, filepath)
        logger.info(f"💾 Model saved to registry: {filepath}")
        return filepath

    def load_model(self, name: str, target_model: torch.nn.Module) -> Dict[str, Any]:
        filepath = os.path.join(self.registry_dir, f"{name}.pt") if not name.endswith(".pt") else name
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found in registry.")

        checkpoint = torch.load(filepath, map_location="cpu")
        target_model.load_state_dict(checkpoint["model_state"], strict=False)
        logger.info(f"📂 Loaded pre-trained weights from {filepath}")
        return checkpoint.get("metadata", {})


class TransferLearningManager:
    """
    Manages Transfer Learning and Topology Generalization across different network fabrics.
    Implements feature layer freezing, layer adaptation, and accelerated fine-tuning.
    """
    def __init__(self, registry: Optional[PretrainedModelRegistry] = None):
        self.registry = registry or PretrainedModelRegistry()

    def transfer_and_fine_tune(
        self,
        source_model: torch.nn.Module,
        target_controller: SDNController,
        fine_tune_episodes: int = 5,
        freeze_features: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Transfers learned feature representations to target_controller, freezes early layers,
        and fine-tunes decision layers on the new topology.
        """
        target_agent = target_controller.agent

        # 1. Transfer compatible layer weights (shared feature representation)
        src_dict = source_model.state_dict()
        target_dict = target_agent.policy_net.state_dict()

        # Copy matching parameter tensors
        transferred_keys = []
        for k, v in src_dict.items():
            if k in target_dict and target_dict[k].shape == v.shape:
                target_dict[k].copy_(v)
                transferred_keys.append(k)

        target_agent.policy_net.load_state_dict(target_dict)
        target_agent.target_net.load_state_dict(target_dict)
        logger.info(f"🔄 Transferred {len(transferred_keys)} parameter tensors to target policy network")

        # 2. Freeze feature extraction layers if requested
        if freeze_features and hasattr(target_agent.policy_net, "freeze_feature_layers"):
            target_agent.policy_net.freeze_feature_layers()
            # Recreate optimizer for trainable parameters only
            trainable_params = [p for p in target_agent.policy_net.parameters() if p.requires_grad]
            if trainable_params:
                target_agent.optimizer = torch.optim.Adam(trainable_params, lr=target_agent.lr)
            logger.info("❄️ Early feature extraction layers FROZEN. Fine-tuning output heads only.")

        # 3. Fine-tune on target network topology
        history = target_controller.train_episodes(num_episodes=fine_tune_episodes, verbose=verbose)

        # 4. Compute transfer efficiency metrics
        initial_reward = history[0]["total_reward"] if history else 0.0
        final_reward = history[-1]["total_reward"] if history else 0.0
        avg_throughput = float(np.mean([h["avg_throughput_mbps"] for h in history])) if history else 0.0

        metrics = {
            "fine_tune_episodes": fine_tune_episodes,
            "transferred_parameters_count": len(transferred_keys),
            "feature_layers_frozen": freeze_features,
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "reward_gain": final_reward - initial_reward,
            "avg_throughput_mbps": avg_throughput,
            "history": history
        }

        return metrics
