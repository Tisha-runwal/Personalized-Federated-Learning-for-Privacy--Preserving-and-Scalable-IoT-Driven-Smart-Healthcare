# PFL-HCare Setup Guide

Complete guide for setting up, training, testing, and visualizing the Personalized Federated Learning framework for IoT-Driven Smart Healthcare.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Datasets](#3-datasets)
4. [Training](#4-training)
5. [Dashboard](#5-dashboard)
6. [Docker Mode](#6-docker-mode)
7. [Configuration](#7-configuration)
8. [Module Tests](#8-module-tests)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Requirements

| Requirement | Value | Notes |
|---|---|---|
| **Python** | 3.10+ | Type hints, torch compatibility |
| **Node.js** | 18+ | For React dashboard |
| **RAM** | 4 GB+ | 8 GB recommended for larger client counts |
| **Disk** | ~2 GB | Dependencies (~1.5 GB), datasets (~50 MB), node_modules (~300 MB) |
| **GPU** | Optional | CPU works fine for 5–10 clients. GPU recommended for 50+ clients |
| **Docker** | Optional | Only needed for Docker simulation mode |

### CPU vs GPU

Unlike many FL projects, PFL-HCare runs well on CPU because the models are intentionally small (~15K and ~52K parameters) to simulate IoT edge devices. GPU provides a speedup for larger client counts:

| Clients | CPU (M-series Mac) | GPU (GTX 1650+) |
|---|---|---|
| 5 clients × 30 rounds | ~4 seconds | ~2 seconds |
| 10 clients × 100 rounds | ~30 seconds | ~10 seconds |
| 50 clients × 200 rounds | ~10 minutes | ~2 minutes |

---

## 2. Environment Setup

### Step 1: Clone and navigate to project

```bash
cd Personalized_Federated_Learning
```

### Step 2: Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: PyTorch, Flower, Opacus, FastAPI, uvicorn, scikit-learn, pandas, and other ML/web dependencies (~1.5 GB total).

### Step 4: Install dashboard dependencies

```bash
cd client
npm install
cd ..
```

### Step 5: Verify installation

```bash
# Python packages
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import flwr; print(f'Flower {flwr.__version__}')"
python3 -c "from server.main import app; print(f'FastAPI app: {app.title}')"

# Node packages
cd client && npx vite --version && cd ..

# Run quick smoke test
python3 -m pytest tests/test_e2e.py -v
```

Expected output:
```
PyTorch 2.x.x
Flower 1.x.x
FastAPI app: PFL-HCare API
vite/5.x.x
tests/test_e2e.py::test_e2e_fedavg_smoke PASSED
tests/test_e2e.py::test_e2e_pfl_hcare_smoke PASSED
```

---

## 3. Datasets

PFL-HCare uses a cascading dataset system with automatic fallback:

```
Tier 1: MIMIC-III Full    → requires PhysioNet credentials
   ↓ not available?
Tier 2: MIMIC-III Demo    → ~100 patients, free download
   ↓ not available?
Tier 3: Heart Disease UCI → 303 samples, auto-download
   ↓ not available?
Tier 4: Synthetic Medical → always available, generated on the fly
```

### UCI HAR (Activity Recognition)

Auto-downloaded on first use:

```bash
python3 scripts/download_data.py har
```

| Property | Value |
|---|---|
| Samples | 10,299 (7,352 train / 2,947 test) |
| Features | 561 (accelerometer + gyroscope signals) |
| Classes | 6 (walking, walking upstairs/downstairs, sitting, standing, laying) |
| IoT Narrative | Wearable sensor data from 30 subjects |

### Medical Dataset (4-tier fallback)

```bash
# Download Heart Disease UCI (Tier 3 fallback)
python3 scripts/download_data.py heart

# Or let it auto-fallback to Synthetic (Tier 4) — no download needed
```

| Tier | Dataset | Samples | Features | Access |
|---|---|---|---|---|
| 1 | MIMIC-III Full | 48,520 | 17 | PhysioNet credentials required |
| 2 | MIMIC-III Demo | ~100 | 17 | Free PhysioNet download |
| 3 | Heart Disease UCI | 303 | 13 | Auto-download |
| 4 | Synthetic Medical | 2,000 (configurable) | 13 | Always available |

The synthetic generator creates realistic vital signs data across 3 patient clusters (healthy 60%, at-risk 30%, critical 10%) with features: heart_rate, systolic_bp, diastolic_bp, spo2, temperature, respiratory_rate, age, bmi, glucose, cholesterol, creatinine, hemoglobin, wbc_count.

### Non-IID Data Partitioning

Data is split across clients using Dirichlet distribution (α parameter controls heterogeneity):

| α Value | Distribution | Use Case |
|---|---|---|
| 0.1 | Extreme non-IID (1–2 dominant classes per client) | Stress test |
| 0.5 | Realistic (uneven but overlapping) | **Default** |
| 5.0 | Mild non-IID (nearly uniform) | Best-case scenario |
| 100 | Approximately IID | Ablation baseline |

---

## 4. Training

### Quick run (CLI)

```bash
# FedAvg baseline (fast, stable)
python3 scripts/run_local.py --method fedavg --rounds 30 --clients 5 --dataset mimic

# PFL-HCare full pipeline (MAML + DP + quantization + adaptive selection)
python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5 --dataset mimic

# All five methods for comparison
for method in fedavg fedprox per_fedavg pfedme pfl_hcare; do
    python3 scripts/run_local.py --method $method --rounds 30 --clients 5 --dataset mimic
done
```

### CLI options

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Path to configuration file |
| `--method` | `pfl_hcare` | FL method: `fedavg`, `fedprox`, `per_fedavg`, `pfedme`, `pfl_hcare` |
| `--rounds` | (from config) | Number of FL communication rounds |
| `--clients` | (from config) | Number of federated clients |
| `--dataset` | (from config) | Dataset: `har` or `mimic` |
| `--lr` | (from config) | Learning rate override |
| `--seed` | (from config) | Random seed for reproducibility |

### How to tell if training is working

| Metric | Bad | Decent | Good |
|---|---|---|---|
| `accuracy` | < 0.60 or oscillating wildly | 0.70–0.85 | > 0.85, stable |
| `loss` | NaN or increasing | Decreasing slowly | < 0.5 and stable |
| `grad_norm` | NaN or > 100 | 0.1 – 10 | 0.1 – 1.0, stable |

### Method-specific behavior

| Method | Expected Behavior |
|---|---|
| **FedAvg** | Smooth convergence to ~88%, stable loss ~0.4 |
| **FedProx** | Similar to FedAvg, slightly more stable on non-IID data |
| **Per-FedAvg** | Faster initial convergence, may oscillate slightly |
| **pFedMe** | Strong personalization, per-client accuracy varies more |
| **PFL-HCare** | Variance due to DP noise — accuracy oscillates but trends upward. This is **expected** and demonstrates the privacy-accuracy tradeoff |

---

## 5. Dashboard

### Starting the dashboard

Three components must run simultaneously:

```bash
# Terminal 1: FastAPI backend
uvicorn server.main:app --port 8000

# Terminal 2: React frontend
cd client && npm run dev

# Terminal 3: FL simulation (triggers live updates to dashboard)
python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5
```

Or start backend and frontend in background:

```bash
nohup uvicorn server.main:app --port 8000 > /tmp/pfl-api.log 2>&1 &
cd client && nohup npm run dev > /tmp/pfl-dashboard.log 2>&1 &
```

Open **http://localhost:5173** in your browser.

### Dashboard views

| View | What It Shows |
|---|---|
| **Overview** | 4 KPI cards (Accuracy, Loss, Privacy Budget, Bandwidth Saved) + Activity Feed |
| **Convergence** | Multi-line accuracy-vs-rounds chart for all methods (PFL-HCare bold, baselines dashed) |
| **Privacy** | Epsilon budget gauge (green→red as budget depletes) + encryption status |
| **Communication** | Bandwidth per round (original vs quantized bars) + cumulative savings |
| **Comparison** | Feature matrix (checkmark grid) + accuracy results per method |

### Control Ribbon

The top ribbon lets you configure and launch training directly from the browser:
- Dataset selector (HAR / Medical)
- Method selector (all 5)
- Number of clients, rounds, noise σ, k-bits
- Start / Stop buttons

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/training/start` | POST | Start training with config JSON |
| `POST /api/training/stop` | POST | Stop current run |
| `GET /api/training/status` | GET | Current status (idle/running/completed) |
| `GET /api/metrics/runs` | GET | List all saved experiment runs |
| `GET /api/metrics/{run_id}` | GET | Get per-round metrics for a run |
| `GET /api/datasets/info` | GET | Available datasets + active tier |
| `POST /api/datasets/partition-preview` | POST | Preview non-IID partition distribution |
| `ws://localhost:8000/ws/live` | WebSocket | Live metric streaming to dashboard |

---

## 6. Docker Mode

### Prerequisites

- Docker Desktop installed and running

### Launch

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- **api** container (port 8000) — FastAPI backend
- **dashboard** container (port 3000) — React frontend via nginx

Open **http://localhost:3000** to view the dashboard.

### Stop

```bash
docker-compose -f docker/docker-compose.yml down
```

### Architecture

```
┌──────────────┐     ┌──────────────┐
│  dashboard   │────►│     api      │
│  (nginx)     │     │  (uvicorn)   │
│  port 3000   │     │  port 8000   │
└──────────────┘     └──────────────┘
     nginx proxies /api/* and /ws/* to api container
```

---

## 7. Configuration

### configs/default.yaml

```yaml
training:
  learning_rate: 0.01        # SGD learning rate
  batch_size: 32             # Per-client batch size
  num_clients: 10            # Number of federated clients
  num_rounds: 200            # Communication rounds
  local_epochs: 5            # Local epochs per round
  seed: 42                   # Random seed

maml:
  inner_lr: 0.01             # MAML inner loop learning rate
  inner_steps: 5             # MAML inner loop gradient steps
  second_order: true         # Full MAML vs FOMAML

privacy:
  noise_multiplier: 0.5      # DP noise scale σ (higher = more private, less accurate)
  max_grad_norm: 1.0         # Gradient clipping bound
  delta: 1.0e-5              # DP delta parameter
  target_epsilon: 10.0       # Privacy budget cap

quantization:
  k_bits: 8                  # Quantization bits (2, 4, 8, 16)
  enabled: true              # Toggle quantization

client_selection:
  adaptive: true             # Enable gradient-norm-based selection
  min_participation_interval: 10  # Minimum rounds between forced participation

dataset:
  name: "har"                # "har" or "mimic"
  partition_alpha: 0.5       # Dirichlet α (lower = more non-IID)
  test_fraction: 0.3         # Train/test split ratio
```

### configs/comparison.yaml

```yaml
comparison_run:
  methods: [fedavg, fedprox, per_fedavg, pfedme, pfl_hcare]
  datasets: [har, mimic]
  rounds: 200
  clients: 10
  seeds: [42, 123, 456]      # Multiple seeds for error bars
  mode: sequential
  save_results: true
```

### Key tuning parameters

| Want to... | Change | From → To |
|---|---|---|
| More privacy | `noise_multiplier` | 0.5 → 1.0+ |
| More accuracy | `noise_multiplier` | 0.5 → 0.1 |
| More compression | `k_bits` | 8 → 2 |
| Less compression error | `k_bits` | 8 → 16 |
| More non-IID | `partition_alpha` | 0.5 → 0.1 |
| More IID | `partition_alpha` | 0.5 → 10.0 |
| Faster training | `local_epochs` | 5 → 1, `num_rounds` ↓ |
| Better convergence | `num_rounds` | 30 → 200 |

---

## 8. Module Tests

### Full test suite

```bash
python3 -m pytest tests/ -v --ignore=tests/test_har_loader.py
```

Expected: **49 passed** (~2 seconds)

> `test_har_loader.py` is excluded because it downloads the UCI HAR dataset, which may fail due to SSL certificate configuration on some systems.

### Individual test files

```bash
python3 -m pytest tests/test_models.py -v          # Model forward pass, param count, gradients
python3 -m pytest tests/test_maml.py -v             # MAML inner/outer loop, 2nd-order mode
python3 -m pytest tests/test_dp.py -v               # DP noise, clipping, epsilon tracking
python3 -m pytest tests/test_quantization.py -v     # k-bit quantize/dequantize, bandwidth
python3 -m pytest tests/test_secure_agg.py -v       # Encrypt/decrypt, aggregation
python3 -m pytest tests/test_partition.py -v         # Dirichlet partitioning, non-IID, heterogeneity
python3 -m pytest tests/test_collector.py -v         # Metrics collection, callbacks, JSON export
python3 -m pytest tests/test_strategies.py -v        # All 5 strategy instantiation
python3 -m pytest tests/test_e2e.py -v               # End-to-end smoke tests (FedAvg + PFL-HCare)
```

### End-to-end smoke test

The quickest way to verify the entire pipeline works:

```bash
python3 -m pytest tests/test_e2e.py -v
```

This runs 2 rounds of FedAvg and PFL-HCare on synthetic data — takes ~2 seconds.

---

## 9. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flwr` | Dependencies not installed | `pip install -r requirements.txt` |
| `ModuleNotFoundError: pfl_hcare` | Wrong working directory | `cd` to project root |
| `SSL: CERTIFICATE_VERIFY_FAILED` (HAR download) | macOS SSL certs not installed | Run `/Applications/Python 3.x/Install Certificates.command` |
| `npm: command not found` | Node.js not installed | Install from [nodejs.org](https://nodejs.org/) |
| `CORS error` in dashboard | API not running | Start `uvicorn server.main:app --port 8000` |
| `WebSocket connection failed` | API not running or wrong port | Verify API is on port 8000 |
| Dashboard shows no data | No simulation running | Run `python3 scripts/run_local.py` or click Start in dashboard |
| Accuracy stuck at 60% | Majority class baseline (60% healthy in synthetic data) | Run more rounds, reduce `noise_multiplier` |
| NaN in loss/gradients | MAML instability with high DP noise | Reduce `noise_multiplier` to 0.1, or use `fedavg` method |
| `docker-compose` not found | Docker Desktop not installed | Install from [docker.com](https://www.docker.com/) |
| Port 8000 already in use | Previous server still running | `lsof -i :8000` then `kill <PID>` |
| Port 5173 already in use | Previous Vite server running | `lsof -i :5173` then `kill <PID>` |

### Key log files

| File | Content |
|---|---|
| `/tmp/pfl-api.log` | FastAPI backend logs (if started with nohup) |
| `/tmp/pfl-dashboard.log` | Vite dev server logs (if started with nohup) |
| `results.db` | SQLite database with all experiment results |

### Getting help

```bash
# Check API health
curl http://localhost:8000/api/training/status

# Check dashboard health
curl -s -o /dev/null -w "HTTP %{http_code}" http://localhost:5173

# View recent run results
python3 -c "
import asyncio, json
from server.db import list_runs, init_db
async def main():
    await init_db()
    runs = await list_runs()
    for r in runs:
        print(f'Run {r[\"id\"]}: {r[\"method\"]} on {r[\"dataset\"]} — {r[\"created_at\"]}')
asyncio.run(main())
"
```
