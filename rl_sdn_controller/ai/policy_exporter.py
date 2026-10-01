import os
import torch
import logging

logger = logging.getLogger(__name__)


def export_policy_to_onnx(model: torch.nn.Module, state_dim: int, output_path: str = "model.onnx"):
    """
    Exports trained PyTorch DQN routing model to ONNX format for C++ / ONNX Runtime edge inference.
    """
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['network_telemetry_state'],
        output_names=['q_values'],
        dynamic_axes={
            'network_telemetry_state': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    logger.info(f"Successfully exported PyTorch model to ONNX: {output_path}")
    return output_path


def export_policy_to_torchscript(model: torch.nn.Module, state_dim: int, output_path: str = "model.pt"):
    """Exports model to TorchScript format."""
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32, device=device)
    traced_model = torch.jit.trace(model, dummy_input)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    traced_model.save(output_path)
    logger.info(f"Successfully exported model to TorchScript: {output_path}")
    return output_path
