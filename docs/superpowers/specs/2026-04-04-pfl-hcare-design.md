# PFL-HCare: Personalized Federated Learning for IoT-Driven Smart Healthcare

**Date:** 2026-04-04
**Type:** Implementation Design Spec
**Paper:** "Personalized Federated Learning for Privacy-Preserving and Scalable IoT-Driven Smart Healthcare" (ICICI-2025)
**Goal:** Demonstration/showcase with polished real-time web dashboard

---

## 1. Overview

Implement the PFL-HCare framework as described in the research paper — a Personalized Federated Learning system combining MAML-based meta-learning, differential privacy, simulated secure aggregation, gradient quantization, and adaptive client selection. The implementation includes a full-stack web dashboard (FastAPI + React) for real-time visualization and comparison against 4 FL baselines.

### Success Criteria

- Full PFL-HCare training pipeline runs end-to-end on UCI HAR + medical fallback dataset
- Dashboard shows live convergence, privacy, communication, and client metrics
- Comparison mode runs all 5 methods and overlays results
- Real differential privacy with (epsilon, delta) accounting via Opacus
- Docker-compose option for multi-container simulation
- Results reproducible via seeded configs

---

## 2. Project Structure

```
Personalized_Federated_Learning/
├── pfl_hcare/                        # Core ML library
│   ├── __init__.py
│   ├── models/                       # Neural network architectures
│   │   ├── __init__.py
│   │   ├── health_classifier.py      # MLP for MIMIC-III / medical fallback
│   │   └── har_classifier.py         # 1D-CNN for UCI HAR
│   ├── fl/                           # Federated learning core
│   │   ├── __init__.py
│   │   ├── server.py                 # Flower server + custom PFL-HCare strategy
│   │   ├── client.py                 # Flower client with MAML fine-tuning
│   │   ├── strategies/               # One strategy per method
│   │   │   ├── __init__.py
│   │   │   ├── fedavg.py             # Vanilla FedAvg
│   │   │   ├── fedprox.py            # FedAvg + proximal term
│   │   │   ├── per_fedavg.py         # MAML-based personalization only
│   │   │   ├── pfedme.py             # Moreau envelope personalization
│   │   │   └── pfl_hcare.py          # Full PFL-HCare strategy
│   │   └── aggregation.py            # Weighted aggregation utilities
│   ├── privacy/                      # Privacy mechanisms
│   │   ├── __init__.py
│   │   ├── differential_privacy.py   # Opacus-based DP with RDP accounting
│   │   ├── secure_aggregation.py     # Simulated homomorphic encryption
│   │   └── quantization.py           # k-bit gradient quantization
│   ├── maml/                         # Meta-learning
│   │   ├── __init__.py
│   │   └── maml.py                   # MAML inner/outer loop (FOMAML toggle)
│   └── metrics/                      # Metric collection
│       ├── __init__.py
│       └── collector.py              # Tracks all metrics, emits via callback
├── data/                             # Dataset handling
│   ├── __init__.py
│   ├── mimic_loader.py               # MIMIC-III loader (4-tier fallback)
│   ├── har_loader.py                 # UCI HAR loader (auto-download)
│   ├── synthetic_generator.py        # Tier 4: synthetic medical data
│   └── partition.py                  # Dirichlet-based non-IID partitioning
├── server/                           # FastAPI backend
│   ├── __init__.py
│   ├── main.py                       # App entry point + CORS + lifespan
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── training.py               # Start/stop/configure training runs
│   │   ├── metrics.py                # REST endpoints for historical metrics
│   │   └── datasets.py               # Dataset info + partition preview
│   ├── ws/
│   │   ├── __init__.py
│   │   └── live.py                   # WebSocket streaming of live metrics
│   ├── orchestrator.py               # Manages comparison runs sequentially
│   └── db.py                         # SQLite persistence for run results
├── client/                           # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx       # Navigation sidebar
│   │   │   │   ├── Header.tsx        # Top bar with progress + status
│   │   │   │   └── ControlRibbon.tsx # Config controls + start/stop
│   │   │   ├── views/
│   │   │   │   ├── OverviewView.tsx  # KPI cards + network topology + activity feed
│   │   │   │   ├── ConvergenceView.tsx   # Multi-line accuracy chart + speed bars
│   │   │   │   ├── PrivacyView.tsx       # Budget gauge + tradeoff + threat model
│   │   │   │   ├── CommunicationView.tsx # Bandwidth bars + heatmap + scalability
│   │   │   │   └── ComparisonView.tsx    # Tables II-IV + experiment history
│   │   │   ├── charts/
│   │   │   │   ├── ConvergenceChart.tsx  # Recharts multi-line
│   │   │   │   ├── PrivacyGauge.tsx      # Epsilon budget gauge
│   │   │   │   ├── TradeoffChart.tsx     # Privacy-accuracy scatter
│   │   │   │   ├── BandwidthChart.tsx    # Stacked bar original vs quantized
│   │   │   │   ├── HeatmapChart.tsx      # Client participation heatmap
│   │   │   │   ├── ScalabilityChart.tsx  # Overhead vs N clients
│   │   │   │   └── SpeedBars.tsx         # Convergence speed horizontal bars
│   │   │   ├── topology/
│   │   │   │   └── NetworkGraph.tsx      # D3 force-directed FL topology
│   │   │   ├── widgets/
│   │   │   │   ├── KpiCard.tsx           # Sparkline KPI card
│   │   │   │   ├── ActivityFeed.tsx      # Scrolling event log
│   │   │   │   ├── ClientCard.tsx        # Individual client status card
│   │   │   │   ├── MethodMatrix.tsx      # Feature comparison matrix
│   │   │   │   ├── ModelFootprint.tsx    # Model size + device compatibility
│   │   │   │   └── EncryptionStatus.tsx  # Lock/unlock animation
│   │   │   └── partition/
│   │   │       └── PartitionPreview.tsx  # Data distribution bar chart
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts       # WebSocket connection + reconnect
│   │   │   └── useTrainingState.ts   # Global training state management
│   │   ├── types/
│   │   │   └── metrics.ts            # TypeScript interfaces for all metrics
│   │   ├── utils/
│   │   │   └── format.ts             # Number formatting, color scales
│   │   ├── App.tsx                   # Router + layout
│   │   └── main.tsx                  # Entry point
│   ├── index.html
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── package.json
├── docker/                           # Docker simulation
│   ├── Dockerfile.fl-server          # FL server container
│   ├── Dockerfile.fl-client          # FL client container
│   ├── Dockerfile.api                # FastAPI backend container
│   ├── Dockerfile.dashboard          # React frontend container
│   └── docker-compose.yml            # Multi-container orchestration
├── configs/
│   ├── default.yaml                  # Default experiment config
│   └── comparison.yaml               # Full comparison run config
├── scripts/
│   ├── run_local.py                  # Single-machine simulation launcher
│   ├── download_data.py              # Dataset download helper
│   └── export_report.py             # Generate PDF/CSV/LaTeX from results
├── tests/
│   ├── test_maml.py
│   ├── test_dp.py
│   ├── test_quantization.py
│   ├── test_strategies.py
│   └── test_partition.py
├── requirements.txt
└── README.md
```

---

## 3. Core Architecture & Data Flow

### Training Pipeline

```
download_data.py → partition.py (Dirichlet α) → Non-IID shards per client
                                                        │
                    ┌───────────────────────────────────▼────────────┐
                    │            Flower Server (strategy)             │
                    │                                                 │
                    │  1. Initialize global model w₀                  │
                    │  2. Select clients (adaptive selection, Eq.9)   │
                    │  3. Broadcast w* to selected clients            │
                    │  4. Collect updates                             │
                    │  5. Apply DP noise to updates (Eq.5)            │
                    │  6. Simulated encryption pass (Eq.6-7)          │
                    │  7. Quantized aggregation (Eq.8)                │
                    │  8. Emit metrics via callback → FastAPI         │
                    │  9. Repeat for T rounds                         │
                    └────┬──────────────────────────────┬────────────┘
                         │                              │
              ┌──────────▼──────┐            ┌──────────▼──────┐
              │   Client i      │    ...     │   Client N      │
              │                 │            │                 │
              │  Receive w*     │            │  Receive w*     │
              │  MAML inner     │            │  MAML inner     │
              │  loop: 3-5      │            │  loop: 3-5      │
              │  gradient steps │            │  gradient steps │
              │  on local Di    │            │  on local DN    │
              │  Return Δw_i    │            │  Return Δw_N    │
              └─────────────────┘            └─────────────────┘
```

### Real-Time Metrics Flow

```
Flower Strategy ──callback──► FastAPI ──WebSocket──► React Dashboard
  (after each round)          (broadcast + SQLite)    (re-render)
```

### Metrics Emitted Per Round

| Metric | Source | Dashboard Location |
|--------|--------|--------------------|
| Global accuracy (train/test) | Server evaluation | Overview KPI + Convergence |
| Per-client accuracy | Each client | Overview topology (color-coded) |
| Per-client loss | Each client | Overview activity feed |
| Privacy budget epsilon spent | RDP accountant | Privacy gauge |
| Accuracy drop from DP | With/without DP delta | Privacy tradeoff chart |
| Bytes transmitted (original) | Pre-quantization size | Communication bandwidth |
| Bytes transmitted (quantized) | Post-quantization size | Communication bandwidth |
| Compression ratio | quantized/original | Communication badge |
| Client selection mask | Adaptive selection | Communication heatmap |
| Client gradient norms | Pre-selection | Overview topology (node size) |
| Round wall-clock time | Timer | Overview activity feed |
| Simulated encryption latency | Secure aggregation | Privacy encryption widget |

---

## 4. Component Specifications

### 4.1 MAML Meta-Learning (`pfl_hcare/maml/maml.py`)

**Outer loop (server-side):**
- Computes meta-initialization w* that generalizes across clients
- Uses Eq.3: w* = argmin Σ F_i(w - α∇F_i(w))

**Inner loop (client-side):**
- Each client fine-tunes w* on local data D_i
- 3-5 gradient steps (configurable)
- Uses Eq.4: w_i = w* - α∇F_i(w*)

**Modes:**
- `second_order=True`: Full MAML with Hessian-vector products via `torch.autograd.grad(create_graph=True)`
- `second_order=False` (FOMAML): First-order approximation, drops second-order terms. Default for HAR CNN model.

### 4.2 Differential Privacy (`pfl_hcare/privacy/differential_privacy.py`)

**Implementation:** Opacus library wrapping client models

**Mechanism:**
- Per-sample gradient clipping (max_grad_norm configurable, default=1.0)
- Gaussian noise injection: w_i' = w_i + N(0, σ²) (Eq.5)
- RDP (Renyi Differential Privacy) accountant tracks cumulative (epsilon, delta)

**Configurable parameters:**
- σ (noise_multiplier): default 0.5, adjustable from dashboard
- delta: default 1e-5
- max_grad_norm: default 1.0
- target_epsilon: optional budget cap — training stops when reached

### 4.3 Simulated Secure Aggregation (`pfl_hcare/privacy/secure_aggregation.py`)

**Not real encryption.** Simulates the workflow for dashboard visualization:

1. Receives model update tensors
2. Serializes to bytes, records original size
3. Adds configurable latency (50-200ms, randomized) to simulate crypto overhead
4. Tags update as "encrypted" in metrics stream
5. After simulated "decryption", passes through unchanged tensors
6. Dashboard shows lock/unlock state transitions with timing

### 4.4 Gradient Quantization (`pfl_hcare/privacy/quantization.py`)

**Real implementation** of k-bit quantization (Eq.8):

```
Q(w_i) = round((w_i - w_min) / (w_max - w_min) * (2^k - 1))
```

- Default k=8 (configurable: 2, 4, 8, 16)
- Tracks original vs quantized byte sizes for bandwidth metrics
- Dequantization on server before aggregation
- Quantization error tracked and reported

### 4.5 Adaptive Client Selection (`pfl_hcare/fl/strategies/pfl_hcare.py`)

**Implements Eq.9:** p_i = ||∇F_i(w)|| / Σ||∇F_j(w)||

- After round 1, server stores gradient norms from each client
- Selection probability proportional to gradient norm
- Clients with higher gradient change (more to learn) participate more often
- Minimum participation floor: each client selected at least once every 10 rounds
- Selection mask emitted as metric for heatmap visualization

### 4.6 Model Architectures

**Health Classifier (`pfl_hcare/models/health_classifier.py`):**
```
Input (features) → Dense(128) + BN + ReLU + Dropout(0.3)
                 → Dense(64) + BN + ReLU + Dropout(0.2)
                 → Dense(32) + ReLU
                 → Output (sigmoid/softmax)
~15K parameters · 61 KB
MAML mode: second_order=True (small model, affordable)
```

**HAR Classifier (`pfl_hcare/models/har_classifier.py`):**
```
Input (9ch × 128 timesteps) → Conv1D(64, k=5) + BN + ReLU + MaxPool
                             → Conv1D(128, k=3) + BN + ReLU + MaxPool
                             → Conv1D(64, k=3) + BN + ReLU
                             → GlobalAvgPool
                             → Dense(64) + ReLU + Dropout(0.3)
                             → Dense(6, softmax)
~52K parameters · 209 KB
MAML mode: second_order=False (FOMAML, CNN too expensive for full MAML)
```

### 4.7 Baseline Strategies

All baselines use the same Flower client/server infrastructure with swapped strategy:

| Method | Strategy Class | Key Implementation Detail |
|--------|---------------|--------------------------|
| FedAvg | `FedAvgStrategy` | `flwr.server.strategy.FedAvg` wrapper, weighted averaging |
| FedProx | `FedProxStrategy` | Adds proximal term `(μ/2)‖w - w_global‖²` to client loss, μ=0.01 |
| Per-FedAvg | `PerFedAvgStrategy` | MAML inner/outer loop, no DP, no quantization |
| pFedMe | `PFedMeStrategy` | Moreau envelope `F_i(w) + (λ/2)‖w - θ_i‖²`, personal params θ_i, λ=15 |
| PFL-HCare | `PFLHCareStrategy` | MAML + Opacus DP + simulated HE + k-bit quantization + adaptive selection |

**Fair comparison guarantees:**
- Same model architecture per dataset
- Same data partitions (same random seed)
- Same number of local epochs (5)
- Same evaluation protocol (test on held-out global test set + per-client local test)

---

## 5. Dataset Handling

### 5.1 UCI HAR (`data/har_loader.py`)

- **Source:** UCI ML Repository (auto-downloaded by `download_data.py`)
- **Size:** 10,299 samples, 561 features (or 9 channels × 128 timesteps for CNN)
- **Classes:** 6 activities (walking, walking_upstairs, walking_downstairs, sitting, standing, laying)
- **Split:** 70% train / 30% test (standard split)
- **IoT narrative:** Represents wearable accelerometer + gyroscope sensors

### 5.2 Medical Dataset — 4-Tier Fallback (`data/mimic_loader.py`)

**Tier 1: MIMIC-III Full**
- 48,520 ICU stays, 17 extracted features, binary mortality prediction
- Requires PhysioNet credentials — `download_data.py` checks for `~/.physionet_credentials`

**Tier 2: MIMIC-III Demo**
- ~100 patients, same schema as full MIMIC-III
- Freely downloadable from PhysioNet without credentials

**Tier 3: Cleveland Heart Disease (UCI)**
- 303 samples, 13 features, binary classification (heart disease presence)
- Auto-downloaded from UCI ML Repository

**Tier 4: Synthetic Medical Generator (`data/synthetic_generator.py`)**
- Generates configurable N patients with realistic vital signs
- Features: heart_rate, systolic_bp, diastolic_bp, spo2, temperature, respiratory_rate, age, bmi, glucose, cholesterol, creatinine, hemoglobin, wbc_count
- 3 patient clusters: healthy (60%), at-risk (30%), critical (10%)
- Non-IID by design: each cluster has distinct distributions
- Deterministic given seed — reproducible

**Auto-detection:** Loader tries each tier in order, logs which is active, emits dataset tier to dashboard.

### 5.3 Non-IID Partitioning (`data/partition.py`)

**Method:** Dirichlet distribution with parameter α

**Presets:**
| Preset | α | Character |
|--------|---|-----------|
| Extreme non-IID | 0.1 | Each client dominated by 1-2 classes |
| Realistic | 0.5 | Uneven but overlapping (default) |
| Mild non-IID | 5.0 | Nearly uniform with slight imbalance |
| Custom | slider | User-defined via dashboard |

**Outputs:**
- Per-client data indices
- Distribution summary (class proportions per client)
- Heterogeneity score (mean pairwise Jensen-Shannon divergence)
- Visualization data for PartitionPreview component

---

## 6. Dashboard Specification

### 6.1 Tech Stack

- React 18 + TypeScript + Vite
- Recharts (line, bar, area, scatter charts)
- D3.js (force-directed network topology on Overview)
- Tailwind CSS (dark theme: slate-900 background)
- Framer Motion (animations: counters, transitions, pulse effects)
- Native WebSocket with auto-reconnect

### 6.2 Visual Design System

- **Theme:** Dark mode, accent gradient blue-500 → cyan-400 for PFL-HCare
- **Font:** Inter for UI, JetBrains Mono for numbers/metrics
- **Colors per method:**
  - PFL-HCare: blue-500 (bold, solid line)
  - pFedMe: purple-400 (dashed)
  - Per-FedAvg: orange-400 (dashed)
  - FedProx: gray-400 (dashed)
  - FedAvg: gray-500 (dotted)
- **Animations:** Number tickers on KPI change, chart line draw-in, pulse on active clients, lock/unlock on encryption, breathing glow on active nodes
- **Responsive:** Optimized for 1920px+ (demo presentations), graceful collapse on smaller

### 6.3 Views

**Sidebar Navigation:** 5 views with icons

**View 1: Overview**
- 4 KPI cards with sparkline trends: Accuracy, Loss, Epsilon Budget, Bandwidth Saved
- D3 force-directed network topology: central server node, orbiting client nodes
  - Node size = data shard size
  - Node color = local accuracy (red → yellow → green gradient)
  - Bright halo = selected this round, dimmed = idle
  - Pulsing lines = data flow animation during round
  - Click node → side drawer with client details
- Activity feed: scrolling log of round events

**View 2: Convergence & Accuracy**
- Large multi-line chart (70% height): accuracy vs rounds for all 5 methods
- Hover tooltip with round #, accuracy, delta from baseline
- Draggable target accuracy threshold line
- Annotation badges: "PFL-HCare hits 90% at round X"
- Dataset toggle: [MIMIC-III] [HAR] with animated transition
- Convergence speed horizontal bars below chart

**View 3: Privacy & Security**
- Epsilon budget gauge (arc/donut): current ε / target ε, percentage used
- Privacy-accuracy tradeoff scatter plot with interactive σ slider
- Encryption status widget: lock/unlock animation, latency display
- Threat model flow animation: Client → [DP Noise] → [Encrypted] → Server → [Decrypted] → [Aggregated]
  - Each stage lights up during the corresponding phase of a round
  - Side panel shows "Attack risk: LOW/MEDIUM" with breakdown

**View 4: Communication & Scalability**
- Stacked bar chart: original vs quantized bytes per round, compression ratio badges
- Cumulative bandwidth savings area chart + big number badge
- Client participation heatmap: rows=clients, cols=rounds, color=selected/idle
- Scalability projection line chart: time-per-round vs N clients for each method

**View 5: Experiment Log & Comparison**
- Live comparison table reproducing Tables II-IV from the paper
  - Tabs: [Accuracy] [Privacy] [Communication]
  - Rows animate as methods finish, winner badges on best values
  - PFL-HCare row highlighted in accent color
- Method feature matrix (checkmark grid)
- Experiment history list: click to replay, compare button to overlay 2 runs
- Export: CSV, PNG, PDF report, LaTeX tables

### 6.4 Control Ribbon

Located below the header, always visible:
- Dataset selector dropdown
- Number of clients (slider: 2-20 for local, 2-100 for docker)
- Noise σ slider (0.1 - 2.0)
- Quantization k dropdown (2, 4, 8, 16 bit)
- Dirichlet α slider (0.1 - 100)
- Number of rounds input
- Learning rate input
- Mode toggle: Local / Docker
- Start Training button (green), Stop button (red)
- Run Full Comparison button (blue)

---

## 7. FastAPI Backend

### 7.1 REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/training/start` | POST | Start training run with config |
| `POST /api/training/stop` | POST | Stop current run |
| `GET /api/training/status` | GET | Current run status |
| `GET /api/metrics/{run_id}` | GET | Historical metrics for a run |
| `GET /api/runs` | GET | List all saved runs |
| `GET /api/datasets/info` | GET | Available datasets + active tier |
| `POST /api/datasets/partition-preview` | POST | Preview partition for given α |
| `POST /api/comparison/start` | POST | Start full comparison run |
| `GET /api/export/{run_id}/{format}` | GET | Export results (csv/json) |

### 7.2 WebSocket

- Endpoint: `ws://localhost:8000/ws/live`
- Broadcasts JSON metrics after each FL round
- Supports multiple concurrent dashboard connections
- Message format:
```json
{
  "type": "round_update",
  "round": 47,
  "method": "pfl_hcare",
  "metrics": {
    "global_accuracy": 0.912,
    "global_loss": 0.38,
    "epsilon_spent": 2.1,
    "bytes_original": 209000,
    "bytes_quantized": 52250,
    "clients_selected": [0, 2, 5, 7, 9],
    "per_client_accuracy": [0.89, 0.91, ...],
    "per_client_gradient_norm": [0.12, 0.45, ...],
    "encryption_latency_ms": 127,
    "round_time_ms": 3400
  }
}
```

### 7.3 Persistence

- SQLite database (`results.db`) stores all run configs + per-round metrics
- Enables replay mode: dashboard can load past runs without re-training
- Schema: `runs` table (id, config, method, dataset, timestamp) + `round_metrics` table (run_id, round, metrics_json)

---

## 8. Docker Setup

### 8.1 Containers

| Container | Image | Role |
|-----------|-------|------|
| `fl-server` | Dockerfile.fl-server | Flower server + strategy |
| `fl-client-1..N` | Dockerfile.fl-client | Flower clients (scaled via replicas) |
| `api` | Dockerfile.api | FastAPI backend |
| `dashboard` | Dockerfile.dashboard | React frontend (nginx) |

### 8.2 docker-compose.yml

- `fl-server`: exposes port 8080 for Flower gRPC
- `fl-client`: scaled via `deploy.replicas: N` (default 5)
- `api`: exposes port 8000, connects to fl-server
- `dashboard`: exposes port 3000, proxies API calls to `api`
- Shared network: `pfl-network`
- Volume mount for configs + data

### 8.3 Usage

```bash
# Local mode (default)
python scripts/run_local.py --config configs/default.yaml

# Docker mode (requires Docker Desktop installed)
docker-compose -f docker/docker-compose.yml up --scale fl-client=10
# If Docker not installed, dashboard shows "Docker not available" with install link
```

---

## 9. Configuration

### default.yaml
```yaml
training:
  learning_rate: 0.01
  batch_size: 32
  num_clients: 10
  num_rounds: 200
  local_epochs: 5
  seed: 42

maml:
  inner_lr: 0.01
  inner_steps: 5
  second_order: true  # false for HAR CNN

privacy:
  noise_multiplier: 0.5
  max_grad_norm: 1.0
  delta: 1.0e-5
  target_epsilon: 10.0

quantization:
  k_bits: 8
  enabled: true

client_selection:
  adaptive: true
  min_participation_interval: 10

secure_aggregation:
  simulated: true
  latency_range_ms: [50, 200]

dataset:
  name: "har"  # or "mimic", "heart", "synthetic"
  partition_alpha: 0.5
  test_fraction: 0.3
```

### comparison.yaml
```yaml
comparison_run:
  methods: [fedavg, fedprox, per_fedavg, pfedme, pfl_hcare]
  datasets: [har, mimic]
  rounds: 200
  clients: 10
  seeds: [42, 123, 456]
  mode: sequential
  save_results: true
```

---

## 10. Reproducibility & Export

- **Seed control:** All random operations (data partition, model init, noise generation, client selection) use configurable seed
- **Config snapshots:** Each run saves its full YAML config to SQLite alongside results
- **Export from dashboard:**
  - CSV: raw per-round metrics
  - PNG: individual chart screenshots (html2canvas)
  - PDF report: auto-generated summary matching paper format (via export_report.py using reportlab)
  - LaTeX tables: formatted Tables II-IV for direct paper inclusion

---

## 11. Dependencies

### Python (requirements.txt)
```
torch>=2.0
flwr>=1.5
opacus>=1.4
fastapi>=0.100
uvicorn>=0.23
websockets>=11.0
pyyaml>=6.0
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
aiosqlite>=0.19
httpx>=0.24
wfdb>=4.1          # PhysioNet data loading
reportlab>=4.0     # PDF export
```

### Node.js (client/package.json)
```
react: ^18
react-dom: ^18
react-router-dom: ^6
recharts: ^2.8
d3: ^7
d3-force: ^3
framer-motion: ^10
tailwindcss: ^3
typescript: ^5
vite: ^5
```

---

## 12. Equations Reference

All equations from the paper, mapped to implementation:

| Eq. | Formula | Implementation Location |
|-----|---------|------------------------|
| 1 | Global FL objective: min Σ (|D_i|/Σ|D_j|) F_i(w) | `pfl_hcare/fl/aggregation.py` |
| 2 | Local loss: F_i(w) = (1/|D_i|) Σ l(w;x,y) | `pfl_hcare/fl/client.py` |
| 3 | MAML meta-objective: w* = argmin Σ F_i(w - α∇F_i(w)) | `pfl_hcare/maml/maml.py` |
| 4 | Client fine-tuning: w_i = w* - α∇F_i(w*) | `pfl_hcare/maml/maml.py` |
| 5 | DP noise: w_i' = w_i + N(0, σ²) | `pfl_hcare/privacy/differential_privacy.py` |
| 6 | HE encryption: Enc(w_i) = w_i^e mod N | `pfl_hcare/privacy/secure_aggregation.py` (simulated) |
| 7 | Secure aggregation: w_t+1 = Σ Dec(Enc(w_i)) | `pfl_hcare/privacy/secure_aggregation.py` (simulated) |
| 8 | Quantization: Q(w) = round((w-w_min)/(w_max-w_min) * (2^k-1)) | `pfl_hcare/privacy/quantization.py` |
| 9 | Client selection: p_i = ||∇F_i||/Σ||∇F_j|| | `pfl_hcare/fl/strategies/pfl_hcare.py` |
