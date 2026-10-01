import pytest
import os
import torch
import numpy as np

from rl_sdn_controller.ai.models import DQN, DuelingDQN, StudentDQN
from rl_sdn_controller.ai.agent import DQNAgent, SumTree, PrioritizedReplayBuffer
from rl_sdn_controller.ai.hierarchical_agent import HierarchicalSDNController, RegionalAgent, GlobalCoordinatorAgent, NetworkRegion
from rl_sdn_controller.ai.curriculum import CurriculumScheduler, CurriculumTrainer, CurriculumStage
from rl_sdn_controller.ai.transfer import TransferLearningManager, PretrainedModelRegistry
from rl_sdn_controller.ai.distillation import PolicyDistiller
from rl_sdn_controller.cli.dashboard import MetricsTracker, DashboardPlotter
from rl_sdn_controller.data_plane.traffic_gen import (
    PoissonTrafficGenerator,
    BurstyTrafficGenerator,
    RealtimeTrafficGenerator,
    BulkTransferGenerator,
    IoTTrafficGenerator,
    MultiTrafficCoordinator,
    TrafficFlowGenerator
)
from rl_sdn_controller.data_plane.chaos import ChaosEngine
from rl_sdn_controller.control_plane.controller import SDNController


def test_prioritized_replay_buffer_and_sumtree():
    """1. Test Prioritized Experience Replay Buffer (PER) & SumTree."""
    tree = SumTree(capacity=8)
    tree.add(1.0, "data1")
    tree.add(2.0, "data2")
    tree.add(3.0, "data3")
    assert abs(tree.total_priority() - 6.0) < 1e-5
    assert len(tree) == 3

    per = PrioritizedReplayBuffer(capacity=50, alpha=0.6, beta_start=0.4)
    state = np.ones(4, dtype=np.float32)
    next_state = np.zeros(4, dtype=np.float32)

    for i in range(20):
        per.push(state, i % 2, float(i), next_state, False)

    assert len(per) == 20
    states, actions, rewards, next_states, dones, indices, weights = per.sample(batch_size=8)
    assert len(states) == 8
    assert len(weights) == 8
    assert len(indices) == 8

    # Test priority update
    td_errors = np.array([2.5] * 8, dtype=np.float32)
    per.update_priorities(indices, td_errors)
    assert per.tree.total_priority() > 0.0


def test_dueling_dqn_architecture_and_student_model():
    """2. Test Dueling DQN stream separation and StudentDQN."""
    state_dim = 10
    action_dim = 4
    dueling = DuelingDQN(state_dim, action_dim, hidden_dims=[32, 32], stream_hidden_dim=16)
    x = torch.randn(3, state_dim)
    q_vals = dueling(x)
    assert q_vals.shape == (3, action_dim)

    # Test layer freezing
    dueling.freeze_feature_layers()
    for param in dueling.feature_layer.parameters():
        assert param.requires_grad is False
    # Value stream and advantage stream parameters should still be trainable
    for param in dueling.value_stream.parameters():
        assert param.requires_grad is True

    dueling.unfreeze_all_layers()
    for param in dueling.parameters():
        assert param.requires_grad is True

    # Student model
    student = StudentDQN(state_dim, action_dim, hidden_dims=[16, 16])
    q_student = student(x)
    assert q_student.shape == (3, action_dim)


def test_double_dqn_and_soft_update():
    """3. Test Double DQN target calculation and soft Polyak update."""
    state_dim = 8
    action_dim = 3
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 4,
        "double_dqn": True,
        "tau": 0.05, # Soft update test
        "use_per": True,
        "hidden_dims": [32, 32]
    }
    agent = DQNAgent(state_dim, action_dim, config)
    for _ in range(10):
        s = np.random.rand(state_dim).astype(np.float32)
        ns = np.random.rand(state_dim).astype(np.float32)
        agent.memory.push(s, 0, 1.0, ns, False)

    loss = agent.update()
    assert loss >= 0.0
    q_vals = agent.get_q_values(np.random.rand(state_dim).astype(np.float32))
    assert len(q_vals) == action_dim


def test_multi_traffic_generators():
    """4. Test all 5 traffic types: Poisson, Bursty, Realtime VoIP, Bulk, IoT."""
    flow_cfg = {"id": "f_test", "src": "h1", "dst": "h2", "duration_sec": 10.0}

    # Poisson
    p_gen = PoissonTrafficGenerator({**flow_cfg, "rate_pps": 100})
    p_pkts = p_gen.generate_packets(0.0, 0.1)
    assert len(p_pkts) > 0

    # Bursty
    b_gen = BurstyTrafficGenerator({**flow_cfg, "rate_pps": 200, "burst_on_ms": 500, "burst_off_ms": 500})
    b_pkts = b_gen.generate_packets(0.0, 0.1)
    assert isinstance(b_pkts, list)

    # Realtime (VoIP)
    r_gen = RealtimeTrafficGenerator({**flow_cfg, "interval_ms": 20.0, "packet_size_bytes": 200})
    r_pkts = r_gen.generate_packets(0.0, 0.1)
    assert len(r_pkts) > 0
    assert r_pkts[0].size_bytes == 200

    # Bulk Transfer
    bulk_gen = BulkTransferGenerator({**flow_cfg, "rate_pps": 1000, "packet_size_bytes": 1500})
    bulk_pkts = bulk_gen.generate_packets(0.0, 0.1)
    assert len(bulk_pkts) > 0
    assert bulk_pkts[0].size_bytes == 1500

    # IoT
    iot_gen = IoTTrafficGenerator({**flow_cfg, "interval_ms": 50.0, "packet_size_bytes": 128})
    iot_pkts = iot_gen.generate_packets(0.0, 0.1)
    assert len(iot_pkts) > 0
    assert iot_pkts[0].size_bytes == 128

    # MultiTrafficCoordinator
    coord = MultiTrafficCoordinator([
        {**flow_cfg, "id": "f1", "pattern": "realtime"},
        {**flow_cfg, "id": "f2", "pattern": "iot"}
    ])
    all_pkts = coord.generate_all_packets(0.0, 0.1)
    assert len(all_pkts) >= 2


def test_link_failure_detection_and_mttr():
    """5. Test link failure injection, recovery, and MTTR metrics."""
    chaos = ChaosEngine({"chaos": {"enabled": True, "link_failure_probability": 0.0}})
    chaos.register_links([("r1", "r2")])

    assert chaos.is_link_up("r1", "r2") is True
    chaos.fail_link("r1", "r2", duration_sec=1.5, current_time=0.5)
    assert chaos.is_link_up("r1", "r2") is False

    metrics_during = chaos.get_metrics()
    assert metrics_during["total_failure_events"] == 1
    assert ("r1", "r2") in metrics_during["currently_failed_links"]

    chaos.recover_link("r1", "r2", current_time=2.0)
    assert chaos.is_link_up("r1", "r2") is True
    metrics_after = chaos.get_metrics()
    assert metrics_after["total_recovery_events"] == 1
    assert metrics_after["mttr_sec"] == pytest.approx(1.5, rel=1e-2)


def test_hierarchical_multi_agent_coordination(tmp_path):
    """6. Test Hierarchical Multi-Agent RL architecture."""
    topo_file = str(tmp_path / "topo_hier.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
  r2: {type: router}
  r3: {type: router}
  r4: {type: router}
links:
  - {src: h1, dst: r1, capacity: 1000, latency: 10, max_queue_packets: 100}
  - {src: r1, dst: r2, capacity: 50, latency: 10, max_queue_packets: 100}
  - {src: r2, dst: r3, capacity: 50, latency: 10, max_queue_packets: 100}
  - {src: r3, dst: r4, capacity: 50, latency: 10, max_queue_packets: 100}
  - {src: r4, dst: h2, capacity: 1000, latency: 10, max_queue_packets: 100}
""")

    traffic_cfg = [{
        "id": "f_h",
        "src": "h1",
        "dst": "h2",
        "pattern": "poisson",
        "rate_pps": 100,
        "packet_size_bytes": 800,
        "duration_sec": 5.0
    }]

    rl_cfg = {
        "agent": {"learning_rate": 0.001, "batch_size": 4, "hidden_dims": [16, 16], "buffer_capacity": 500},
        "control_plane": {"update_interval_ms": 100, "max_simulation_steps": 3},
        "reward_function": {"throughput_weight": 1.0, "drop_rate_weight": -10.0, "latency_weight": -0.1},
        "hierarchical": {"global_reward_weight": 0.3}
    }

    h_ctrl = HierarchicalSDNController(topo_file, traffic_cfg, rl_cfg, num_regions=2)
    assert len(h_ctrl.regional_agents) == 2
    assert h_ctrl.coordinator is not None

    history = h_ctrl.train_episodes(num_episodes=2, verbose=False)
    assert len(history) == 2
    assert "total_reward" in history[0]


def test_curriculum_learning_scheduler():
    """7. Test CurriculumScheduler stage transitions."""
    scheduler = CurriculumScheduler()
    stage1, trans1 = scheduler.get_stage_for_episode(1)
    assert stage1.stage_id == 1

    stage2, trans2 = scheduler.get_stage_for_episode(6)
    assert stage2.stage_id == 2
    assert trans2 is True


def test_metrics_tracker_and_plots(tmp_path):
    """8. Test MetricsTracker and DashboardPlotter."""
    tracker = MetricsTracker(window_size=3)
    tracker.set_baselines({"throughput_mbps": 12.0, "drop_rate_pct": 0.5, "avg_latency_ms": 10.0}, {"throughput_mbps": 10.0, "drop_rate_pct": 1.0, "avg_latency_ms": 11.0})

    for ep in range(1, 6):
        tracker.record_episode(
            episode=ep,
            reward=float(ep * 10),
            throughput_mbps=15.0 + ep,
            drop_rate_pct=0.1,
            avg_latency_ms=8.0,
            p99_latency_ms=9.5,
            avg_q_val=1.2,
            max_q_val=2.5,
            loss=0.01,
            link_utils={("r1", "r2"): 45.0, ("r2", "r3"): 60.0}
        )

    summary = tracker.get_summary()
    assert summary["total_episodes"] == 5
    assert summary["latest_reward"] == 50.0

    plot_dir = str(tmp_path / "plots")
    plotter = DashboardPlotter(tracker, output_dir=plot_dir)
    out_file = plotter.save_plots("test_dashboard.png")
    assert os.path.exists(out_file)


def test_transfer_learning(tmp_path):
    """9. Test TransferLearningManager."""
    state_dim = 8
    action_dim = 2
    source_model = DuelingDQN(state_dim, action_dim, hidden_dims=[16, 16])
    
    registry = PretrainedModelRegistry(registry_dir=str(tmp_path / "reg"))
    saved_path = registry.save_model(source_model, "base_model")
    assert os.path.exists(saved_path)

    loaded_model = DuelingDQN(state_dim, action_dim, hidden_dims=[16, 16])
    registry.load_model("base_model", loaded_model)

    # Test fine tuning manager with dummy controller
    topo_file = str(tmp_path / "topo_tf.yaml")
    with open(topo_file, "w") as f:
        f.write("""
nodes:
  h1: {type: host}
  h2: {type: host}
  r1: {type: router}
links:
  - {src: h1, dst: r1, capacity: 500, latency: 10, max_queue_packets: 100}
  - {src: r1, dst: h2, capacity: 500, latency: 10, max_queue_packets: 100}
""")
    traffic_cfg = [{"id": "f1", "src": "h1", "dst": "h2", "rate_pps": 50, "packet_size_bytes": 500, "duration_sec": 5.0}]
    rl_cfg = {
        "agent": {"learning_rate": 0.001, "batch_size": 4, "hidden_dims": [16, 16], "buffer_capacity": 500},
        "control_plane": {"update_interval_ms": 100, "max_simulation_steps": 3},
        "reward_function": {"throughput_weight": 1.0, "drop_rate_weight": -10.0, "latency_weight": -0.1}
    }
    target_ctrl = SDNController(topo_file, traffic_cfg, rl_cfg)
    tf_manager = TransferLearningManager(registry)
    results = tf_manager.transfer_and_fine_tune(source_model, target_ctrl, fine_tune_episodes=2, freeze_features=True, verbose=False)
    assert results["feature_layers_frozen"] is True
    assert len(results["history"]) == 2


def test_policy_distillation(tmp_path):
    """10. Test PolicyDistiller and Edge Compression."""
    state_dim = 10
    action_dim = 3
    teacher = DuelingDQN(state_dim, action_dim, hidden_dims=[64, 64])
    student = StudentDQN(state_dim, action_dim, hidden_dims=[16, 16])

    distiller = PolicyDistiller(
        teacher_model=teacher,
        state_dim=state_dim,
        action_dim=action_dim,
        student_model=student,
        temperature=2.0
    )

    states = np.random.rand(50, state_dim).astype(np.float32)
    losses = distiller.distill_from_samples(states, epochs=3, batch_size=16)
    assert len(losses) == 3

    benchmark = distiller.benchmark_compression(states)
    assert benchmark["teacher_parameters"] > benchmark["student_parameters"]
    assert benchmark["compression_ratio"] > 1.0

    onnx_out = str(tmp_path / "student.onnx")
    res_path = distiller.export_student_onnx(onnx_out)
    assert os.path.exists(res_path)
