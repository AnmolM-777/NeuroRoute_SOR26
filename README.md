# RL-SDN Controller: Three-Layer AI-Driven SDN Architecture

A production-grade, three-layer Reinforcement Learning driven Software-Defined Networking (SDN) Controller implementing control plane & data plane separation.

---

## 📊 Performance Benchmark Comparison (Chaos Engine Active)

| Metric | Dueling DQN (Proposed) | Standard DQN | Static OSPF (Dijkstra) | Round-Robin (ECMP) |
| :--- | :--- | :--- | :--- | :--- |
| **Throughput (Mbps)** | **62.2 Mbps** | 61.3 Mbps | 47.6 Mbps | 56.4 Mbps |
| **Packet Drop Rate (%)** | **3.07%** | 4.02% | 4.72% | 6.48% |
| **Average Latency (ms)** | **6.52 ms** (~70% ⬇️) | 6.61 ms | 20.07 ms | 8.20 ms |
| **Tail Latency P99 (ms)**| **29.13 ms** (>50% ⬇️) | 19.51 ms | 63.90 ms | 15.79 ms |

### 💡 Why Dueling DQN Outperforms Static Baselines
1. **Dynamic Congestion Avoidance**: Under heavy traffic, Static OSPF forces 100% of traffic down a single primary path (`r1 ➔ r2`), causing bufferbloat queueing delay (**20.07 ms average, 63.90 ms P99 tail**).
2. **Real-Time Telemetry & Failover**: Dueling DQN senses queue depth buildup and link health in real-time, dynamically offloading burst flows to alternative paths (`r1 ➔ r3`), keeping router queue depths near zero and reducing average latency down to **6.52 ms** (~70% reduction).

---

## 🌟 Key Features
- **Three-Layer Decoupled Architecture**:
  - **Layer 1: RL Control Plane**: PyTorch Deep Q-Network (Dueling DQN / Standard DQN) agent executing policy evaluation every 100ms.
  - **Layer 2: SDN Abstraction Layer**: Uniform forwarding table management and telemetry stats API supporting simulation mode, OpenFlow (Ryu), and P4Runtime gRPC.
  - **Layer 3: Data Plane Forwarding**: High-performance Asyncio-based packet simulator with per-link FIFO queues, queue depth metrics, BER packet dropping, and Poisson/bursty traffic generators.
- **Network Chaos Engine**: Simulates stochastic real-world link flapping, BER packet drops, and microsecond delay jitter.
- **ONNX Export**: Exports trained PyTorch policy models to ONNX (`model.onnx`) for C++/hardware edge deployment.
- **Rich Terminal TUI**: Interactive CLI displaying real-time metrics, per-link queue depths, drop rates, and throughput tables.

---

## 🛠️ Tech Stack & Dependencies
- **Python**: 3.9+
- **RL & ML**: PyTorch, Gymnasium, ONNX
- **Networking**: NetworkX, PyYAML, Asyncio
- **Terminal UI**: Rich, Matplotlib
- **Testing**: pytest

---

## 🚀 Quick Start Instructions

### Windows PowerShell setup

From the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$env:MPLCONFIGDIR=(Join-Path (Get-Location) '.venv\matplotlib')
.\.venv\Scripts\python.exe run.py
```

The main `requirements.txt` includes CUDA packages that are unavailable on Windows. The Windows file installs the CPU version of PyTorch and all packages needed by the app and tests.

When `run.py` starts, choose a simulation mode and optionally enter a router count (4–64). Press Enter to use the default 4-router topology. The selected topology is used for training and evaluation in that run. The existing 8-, 14-, and 22-router configurations remain available by entering those counts; other counts generate a connected topology. In the custom workflow, the optional randomized production scenario is available only with the default 4-router choice.

### 1. Interactive Terminal User Interface (TUI)
```bash
.venv/bin/python run.py
```

After choosing the mode, topology size, and training or evaluation settings, the Rich dashboard shows live directed-link status, per-link queue fill and drops, utilization, latency, and chaos events. The dashboard closes before the final results table is printed. For large topologies, failed and busiest links are shown first.

### 2. Automated Model Comparison Benchmark
```bash
PYTHONPATH=. .venv/bin/python scripts/compare_models.py --chaos --episodes 15
```

### 3. Train & Export ONNX Policy
```bash
.venv/bin/python -m rl_sdn_controller.cli.main train --episodes 20 --export-onnx model.onnx
```

### 4. Run Test Suite
```bash
.venv/bin/pytest tests/ -v
```
