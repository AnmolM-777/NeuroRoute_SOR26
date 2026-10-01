import time
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_sdn_controller.ai.models import DQN, DuelingDQN, StudentDQN
from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx

logger = logging.getLogger(__name__)


class PolicyDistiller:
    """
    Policy Knowledge Distillation Engine.
    Compresses high-capacity Teacher DuelingDQN into ultra-lightweight StudentDQN
    for low-latency hardware switch & edge router deployment.
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        state_dim: int,
        action_dim: int,
        student_model: Optional[nn.Module] = None,
        temperature: float = 2.0,
        alpha_distill: float = 0.7,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Freeze teacher model
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Initialize student model (compact architecture)
        if student_model is None:
            self.student = StudentDQN(state_dim, action_dim, hidden_dims=[32, 32]).to(self.device)
        else:
            self.student = student_model.to(self.device)

        self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)

    def distill_step(self, states: torch.Tensor) -> float:
        """
        Performs one gradient descent step on distillation loss:
        L = alpha * T^2 * KL(softmax(Q_student/T), softmax(Q_teacher/T)) + (1 - alpha) * MSE(Q_student, Q_teacher)
        """
        self.student.train()

        with torch.no_grad():
            teacher_q = self.teacher(states)

        student_q = self.student(states)

        # Temperature-scaled soft probability distributions
        p_teacher = F.softmax(teacher_q / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_q / self.temperature, dim=-1)

        # Distillation loss
        kl_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (self.temperature ** 2)
        mse_loss = F.mse_loss(student_q, teacher_q)

        loss = (self.alpha_distill * kl_loss) + ((1.0 - self.alpha_distill) * mse_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def distill_from_samples(self, states_data: np.ndarray, epochs: int = 20, batch_size: int = 64) -> List[float]:
        """Trains student network on a collection of state vectors."""
        num_samples = len(states_data)
        losses = []

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(num_samples)
            epoch_losses = []

            for i in range(0, num_samples, batch_size):
                idx = perm[i:i + batch_size]
                batch_states = torch.FloatTensor(states_data[idx]).to(self.device)
                l = self.distill_step(batch_states)
                epoch_losses.append(l)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(avg_loss)
            if epoch % 5 == 0 or epoch == epochs:
                logger.info(f"🧪 [Distillation] Epoch {epoch}/{epochs} | Loss: {avg_loss:.5f}")

        return losses

    def benchmark_compression(self, test_states: np.ndarray) -> Dict[str, Any]:
        """
        Benchmarks inference speedup, model parameter reduction, and action agreement retention.
        """
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        compression_ratio = float(teacher_params / student_params) if student_params > 0 else 1.0

        test_t = torch.FloatTensor(test_states).to(self.device)

        self.teacher.eval()
        self.student.eval()

        # Measure inference latency (100 runs)
        with torch.no_grad():
            # Warmup
            _ = self.teacher(test_t)
            _ = self.student(test_t)

            # Teacher timing
            t0 = time.perf_counter()
            for _ in range(100):
                t_q = self.teacher(test_t)
            t_teacher = (time.perf_counter() - t0) / 100.0

            # Student timing
            t0 = time.perf_counter()
            for _ in range(100):
                s_q = self.student(test_t)
            t_student = (time.perf_counter() - t0) / 100.0

            # Action agreement calculation
            t_actions = t_q.argmax(dim=-1).cpu().numpy()
            s_actions = s_q.argmax(dim=-1).cpu().numpy()
            agreement_rate = float(np.mean(t_actions == s_actions) * 100.0)

        speedup = float(t_teacher / t_student) if t_student > 0 else 1.0

        metrics = {
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "compression_ratio": round(compression_ratio, 2),
            "teacher_latency_us": round(t_teacher * 1_000_000, 2),
            "student_latency_us": round(t_student * 1_000_000, 2),
            "inference_speedup_x": round(speedup, 2),
            "action_retention_pct": round(agreement_rate, 2)
        }

        return metrics

    def export_student_onnx(self, output_path: str = "model_student.onnx") -> str:
        """Exports the compressed student network to ONNX."""
        return export_policy_to_onnx(self.student, self.state_dim, output_path)
