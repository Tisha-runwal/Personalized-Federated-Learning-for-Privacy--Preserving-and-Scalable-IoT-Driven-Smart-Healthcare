# How to Run PFL-HCare — Step-by-Step Execution Guide

A detailed, beginner-friendly walkthrough for installing, running, and demonstrating every component of the PFL-HCare framework. Follow each step exactly as written.

---

## Table of Contents

1. [Prerequisites — What You Need Before Starting](#1-prerequisites--what-you-need-before-starting)
2. [Installation — From Zero to Ready](#2-installation--from-zero-to-ready)
3. [Verify Everything Works](#3-verify-everything-works)
4. [Run Mode 1: CLI Simulation (No Dashboard)](#4-run-mode-1-cli-simulation-no-dashboard)
5. [Run Mode 2: Full Dashboard Experience](#5-run-mode-2-full-dashboard-experience)
6. [Run Mode 3: Docker Deployment](#6-run-mode-3-docker-deployment)
7. [Run All 5 Methods for Comparison](#7-run-all-5-methods-for-comparison)
8. [Understanding the Output](#8-understanding-the-output)
9. [Customizing Your Run](#9-customizing-your-run)
10. [Stopping Everything](#10-stopping-everything)
11. [Common Issues & Fixes](#11-common-issues--fixes)

---

## 1. Prerequisites — What You Need Before Starting

### Required Software

| Software | Minimum Version | How to Check | How to Install |
|---|---|---|---|
| **Python** | 3.10+ | `python3 --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | 21+ | `pip3 --version` | Comes with Python |
| **Node.js** | 18+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | 9+ | `npm --version` | Comes with Node.js |
| **Git** | 2.0+ | `git --version` | [git-scm.com](https://git-scm.com/) |

### Optional Software

| Software | Purpose | How to Install |
|---|---|---|
| **Docker Desktop** | For containerized deployment (Run Mode 3) | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **NVIDIA GPU + CUDA** | Faster training with 50+ clients | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| Disk Space | 2 GB free | 5 GB free |
| CPU | Any modern processor | Multi-core (parallelizes clients) |
| GPU | Not required | Any CUDA GPU (for large-scale runs) |

---

## 2. Installation — From Zero to Ready

Open a terminal and run these commands **one at a time**, in order:

### Step 1: Navigate to the project directory

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
```

### Step 2: (Recommended) Create a Python virtual environment

This keeps project dependencies isolated from your system Python:

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note:** Every time you open a new terminal to work on this project, run `source venv/bin/activate` first.

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs ~17 packages including PyTorch, Flower, Opacus, FastAPI, and more. It may take 2–5 minutes depending on your internet speed.

**What you should see:**
```
Successfully installed torch-2.x.x flwr-1.x.x opacus-1.x.x fastapi-0.x.x ...
```

### Step 4: Install the React dashboard dependencies

```bash
cd client
npm install
cd ..
```

**What you should see:**
```
added 250+ packages in Xs
```

### Step 5: (Optional) Download the UCI HAR dataset

```bash
python3 scripts/download_data.py har
```

> **Note:** If this fails with an SSL error, don't worry — the system automatically falls back to synthetic data. You can fix SSL later (see Troubleshooting).

---

## 3. Verify Everything Works

Run these verification commands to confirm your installation is complete:

### Check Python packages

```bash
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import flwr; print(f'✓ Flower {flwr.__version__}')"
python3 -c "import opacus; print(f'✓ Opacus {opacus.__version__}')"
python3 -c "from server.main import app; print(f'✓ FastAPI app: {app.title}')"
```

**Expected output:**
```
✓ PyTorch 2.x.x
✓ Flower 1.x.x
✓ Opacus 1.x.x
✓ FastAPI app: PFL-HCare API
```

### Check Node.js

```bash
cd client && npx vite --version && cd ..
```

**Expected output:**
```
vite/5.x.x
```

### Run the smoke test (most important!)

```bash
python3 -m pytest tests/test_e2e.py -v
```

**Expected output:**
```
tests/test_e2e.py::test_e2e_fedavg_smoke PASSED
tests/test_e2e.py::test_e2e_pfl_hcare_smoke PASSED
==================== 2 passed in ~2s ====================
```

If both pass, **your installation is complete and working.**

### Run the full test suite (optional but recommended)

```bash
python3 -m pytest tests/ -v --ignore=tests/test_har_loader.py
```

**Expected:** 49 passed in ~2 seconds.

---

## 4. Run Mode 1: CLI Simulation (No Dashboard)

The simplest way to run — just a single command, see results in terminal.

### Run FedAvg (simplest baseline)

```bash
python3 scripts/run_local.py \
    --method fedavg \
    --rounds 30 \
    --clients 5 \
    --dataset mimic
```

**What you should see:**
```
Starting simulation: method=fedavg  rounds=30  clients=5
Loading dataset: mimic
Loading Synthetic Medical Data (Tier 4)
Partitioning 1400 samples across 5 clients (alpha=0.50)
Round 1 / 30
  Round 1 — accuracy: 0.8840  loss: 0.4515
Round 2 / 30
  Round 2 — accuracy: 0.8613  loss: 0.3058
...
Round 30 / 30
  Round 30 — accuracy: 0.8880  loss: 0.4045

=== Simulation complete ===
  final_loss: 0.4045
  final_accuracy: 0.8880
```

### Run PFL-HCare (full framework with DP)

```bash
python3 scripts/run_local.py \
    --method pfl_hcare \
    --rounds 30 \
    --clients 5 \
    --dataset mimic
```

**What you should see:**
```
Starting simulation: method=pfl_hcare  rounds=30  clients=5
...
Round 5 — accuracy: 0.7267  loss: 54.5810     (learning with DP noise)
Round 13 — accuracy: 0.8193  loss: 2.5562     (converging)
Round 24 — accuracy: 0.8760  loss: 3.4804     (peak accuracy)
...
=== Simulation complete ===
  final_accuracy: 0.67–0.87  (varies due to DP noise — this is expected!)
```

> **Why does PFL-HCare accuracy oscillate?** Because Differential Privacy adds Gaussian noise to every model update. This is the privacy-accuracy tradeoff — the core contribution of the paper. The dashboard's Privacy Panel visualizes this beautifully.

### Run all 5 methods back-to-back

```bash
for method in fedavg fedprox per_fedavg pfedme pfl_hcare; do
    echo ""
    echo "========== Running $method =========="
    python3 scripts/run_local.py --method $method --rounds 20 --clients 5 --dataset mimic
done
```

### All CLI options

```bash
python3 scripts/run_local.py --help
```

| Flag | Default | Description | Example |
|---|---|---|---|
| `--method` | `pfl_hcare` | FL method to run | `--method fedavg` |
| `--rounds` | 200 (from config) | Number of FL rounds | `--rounds 50` |
| `--clients` | 10 (from config) | Number of federated clients | `--clients 5` |
| `--dataset` | `har` (from config) | Dataset to use | `--dataset mimic` |
| `--lr` | 0.01 (from config) | Learning rate | `--lr 0.001` |
| `--seed` | 42 (from config) | Random seed | `--seed 123` |
| `--config` | `configs/default.yaml` | Config file path | `--config configs/comparison.yaml` |

---

## 5. Run Mode 2: Full Dashboard Experience

The showcase mode — real-time visualization in your browser.

### Step 1: Start the FastAPI backend (Terminal 1)

Open a terminal and run:

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
source venv/bin/activate   # if using venv
uvicorn server.main:app --port 8000
```

**What you should see:**
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Leave this terminal running.** Do not close it.

### Step 2: Start the React dashboard (Terminal 2)

Open a **new** terminal and run:

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning/client
source ../venv/bin/activate   # if using venv
npm run dev
```

**What you should see:**
```
  VITE v5.x.x  ready in XXXms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Leave this terminal running.** Do not close it.

### Step 3: Open the dashboard in your browser

Open your web browser and go to:

```
http://localhost:5173
```

You should see the **PFL-HCare Dashboard** with:
- A dark-themed interface
- Left sidebar with 5 navigation icons
- Top control ribbon with configuration options
- Main content area (will show data once training starts)

### Step 4: Start a training run (Terminal 3)

Open a **third** terminal and run:

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
source venv/bin/activate   # if using venv
python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5 --dataset mimic
```

**Now watch the dashboard!** You should see:
- **Overview** tab: KPI cards updating with accuracy, loss, epsilon, bandwidth
- **Activity Feed**: Scrolling log of round-by-round events
- Metrics streaming in real-time via WebSocket

### Alternative: Start training from the dashboard UI

Instead of Terminal 3, you can also start training directly from the browser:

1. In the **Control Ribbon** at the top of the dashboard:
   - Select **Dataset**: `Medical` or `HAR`
   - Select **Method**: `PFL-HCare` (or any of the 5)
   - Set **Clients**: 5
   - Set **Rounds**: 30
2. Click the green **Start Training** button
3. Watch the metrics appear in real-time

### Background mode (one-liner)

If you don't want 3 terminals, start backend + frontend in the background:

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning

# Start both servers in background
nohup uvicorn server.main:app --port 8000 > /tmp/pfl-api.log 2>&1 &
echo "API started (PID: $!)"

cd client && nohup npm run dev > /tmp/pfl-dashboard.log 2>&1 &
echo "Dashboard started (PID: $!)"
cd ..

# Wait for servers to boot
sleep 3

# Verify they're running
curl -s http://localhost:8000/api/training/status
echo ""
curl -s -o /dev/null -w "Dashboard: HTTP %{http_code}" http://localhost:5173
echo ""

echo ""
echo "✓ Open http://localhost:5173 in your browser"
echo "✓ Then run: python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5 --dataset mimic"
```

---

## 6. Run Mode 3: Docker Deployment

For a clean, reproducible deployment without installing anything on your host machine (except Docker).

### Prerequisites

- Docker Desktop installed and running
- Verify: `docker --version` and `docker-compose --version`

### Step 1: Build and start containers

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
docker-compose -f docker/docker-compose.yml up --build
```

This builds 2 containers:
- **api** — Python + FastAPI (port 8000)
- **dashboard** — Node.js build + nginx (port 3000)

First build takes 3–5 minutes. Subsequent runs are cached.

**What you should see:**
```
Creating network "docker_pfl-network" with driver "bridge"
Building api ...
Building dashboard ...
...
api_1        | INFO: Uvicorn running on http://0.0.0.0:8000
dashboard_1  | /docker-entrypoint.sh: Configuration complete; ready for start up
```

### Step 2: Open the dashboard

```
http://localhost:3000
```

> **Note:** In Docker mode, the dashboard runs on port **3000** (not 5173).

### Step 3: Stop containers

```bash
docker-compose -f docker/docker-compose.yml down
```

---

## 7. Run All 5 Methods for Comparison

### Quick comparison script

Create and run a comparison across all methods:

```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
source venv/bin/activate

echo "╔══════════════════════════════════════════════╗"
echo "║  PFL-HCare — 5-Method Comparison Run        ║"
echo "╚══════════════════════════════════════════════╝"

for method in fedavg fedprox per_fedavg pfedme pfl_hcare; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $method"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 scripts/run_local.py \
        --method $method \
        --rounds 30 \
        --clients 5 \
        --dataset mimic \
        --seed 42
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All 5 methods completed!                    ║"
echo "╚══════════════════════════════════════════════╝"
```

### Expected results (approximate)

| Method | Final Accuracy | Convergence | Notes |
|---|---|---|---|
| **FedAvg** | ~88.8% | Smooth, stable | Baseline — no personalization or privacy |
| **FedProx** | ~88.5% | Slightly more stable | Proximal term prevents client drift |
| **Per-FedAvg** | ~87-89% | Similar to FedAvg | Personalization regularizer active |
| **pFedMe** | ~85-88% | Strong personalization | Moreau envelope — per-client params diverge |
| **PFL-HCare** | ~67-87% | Oscillates (DP noise) | Full pipeline — accuracy varies due to privacy |

> **Why does PFL-HCare have variable accuracy?** The Differential Privacy mechanism intentionally injects Gaussian noise into model updates (Eq.5 in the paper). Higher noise = stronger privacy guarantees but more accuracy variance. This is the fundamental privacy-accuracy tradeoff. Reduce `--noise_multiplier` via config to see less variance.

---

## 8. Understanding the Output

### Terminal output format

Each round prints:

```
Round 5 / 30
  Round 5 — accuracy: 0.7267  loss: 54.5810
  [metrics] round=5  method=pfl_hcare  global_accuracy=0.7267  global_loss=54.5810  num_clients=5  avg_grad_norm=55.7255  per_client_accuracy=[0.727, 0.727, 0.727, 0.727, 0.727]
```

| Field | Meaning |
|---|---|
| `accuracy` | Global test accuracy (averaged across all clients' test data) |
| `loss` | Global test loss (cross-entropy) |
| `num_clients` | Number of clients participating |
| `avg_grad_norm` | Average L2 norm of parameter updates (indicates learning magnitude) |
| `per_client_accuracy` | Individual accuracy per client (shows personalization) |

### Final summary

```
=== Simulation complete ===
  final_loss: 0.4045
  final_accuracy: 0.8880
  rounds_completed: 30
  method: fedavg
```

### Dashboard views explained

| View | What to Look For |
|---|---|
| **Overview** | Are KPI cards updating? Green = good accuracy, red = declining |
| **Convergence** | Is the accuracy line trending upward? PFL-HCare should be bold blue |
| **Privacy** | Epsilon gauge filling up = privacy budget being consumed |
| **Communication** | Quantized bars should be shorter than original bars (compression working) |
| **Comparison** | Feature checkmark matrix shows what each method includes |

### Stored results

Every training run is saved to SQLite:

```bash
# List all saved runs
python3 -c "
import asyncio
from server.db import list_runs, init_db
async def main():
    await init_db()
    runs = await list_runs()
    for r in runs:
        print(f'  Run #{r[\"id\"]}: {r[\"method\"]} on {r[\"dataset\"]} — {r[\"created_at\"]}')
asyncio.run(main())
"
```

### API endpoints for programmatic access

```bash
# Get training status
curl http://localhost:8000/api/training/status

# List all experiment runs
curl http://localhost:8000/api/metrics/runs

# Get metrics for run #1
curl http://localhost:8000/api/metrics/1

# Get dataset info (which tier is active)
curl http://localhost:8000/api/datasets/info

# Preview data partition for alpha=0.1
curl -X POST http://localhost:8000/api/datasets/partition-preview \
    -H "Content-Type: application/json" \
    -d '{"num_clients": 5, "alpha": 0.1, "seed": 42}'
```

---

## 9. Customizing Your Run

### Change privacy level

Edit `configs/default.yaml` or use CLI:

```bash
# Low privacy, high accuracy (σ = 0.1)
python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5 --dataset mimic

# High privacy, lower accuracy (σ = 2.0) — edit config:
# In configs/default.yaml, change:  noise_multiplier: 2.0
```

### Change data heterogeneity

```bash
# Very non-IID (each client sees mostly 1 class)
# In configs/default.yaml, change:  partition_alpha: 0.1

# Nearly IID (all clients see similar distribution)
# In configs/default.yaml, change:  partition_alpha: 100.0
```

### Change compression level

```bash
# Maximum compression (93.75% bandwidth savings, more error)
# In configs/default.yaml, change:  k_bits: 2

# Minimum compression (less savings, minimal error)
# In configs/default.yaml, change:  k_bits: 16
```

### Scale up

```bash
# More clients (simulates larger IoT network)
python3 scripts/run_local.py --method pfl_hcare --rounds 100 --clients 20 --dataset mimic

# More rounds (better convergence)
python3 scripts/run_local.py --method pfl_hcare --rounds 200 --clients 10 --dataset mimic
```

### Use UCI HAR dataset (activity recognition)

```bash
# First download (one-time)
python3 scripts/download_data.py har

# Then run with HAR
python3 scripts/run_local.py --method fedavg --rounds 30 --clients 5 --dataset har
```

---

## 10. Stopping Everything

### Stop CLI simulation

Press `Ctrl+C` in the terminal running the simulation.

### Stop dashboard servers

If running in foreground (separate terminals):
- Press `Ctrl+C` in each terminal

If running in background:
```bash
# Find and kill API server
lsof -i :8000 | grep LISTEN
kill <PID>

# Find and kill Vite dashboard
lsof -i :5173 | grep LISTEN
kill <PID>

# Or kill all at once
pkill -f "uvicorn server.main"
pkill -f "vite"
```

### Stop Docker

```bash
docker-compose -f docker/docker-compose.yml down
```

---

## 11. Common Issues & Fixes

### Installation Issues

| Problem | Solution |
|---|---|
| `pip install` fails with permission error | Use `pip install --user -r requirements.txt` or activate venv first |
| `npm install` fails | Delete `client/node_modules` and `client/package-lock.json`, then retry |
| PyTorch installation is very slow | Normal — PyTorch is ~800 MB. Be patient |
| `command not found: python3` | Install Python from [python.org](https://www.python.org/downloads/) |
| `command not found: node` | Install Node.js from [nodejs.org](https://nodejs.org/) |

### Runtime Issues

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: pfl_hcare` | Make sure you're in the project root directory |
| `ModuleNotFoundError: flwr` | Run `pip install -r requirements.txt` |
| `SSL: CERTIFICATE_VERIFY_FAILED` when downloading HAR | Run: `/Applications/Python\ 3.14/Install\ Certificates.command` (macOS) |
| Port 8000 already in use | `lsof -i :8000` → `kill <PID>`, then retry |
| Port 5173 already in use | `lsof -i :5173` → `kill <PID>`, then retry |
| Dashboard shows blank page | Check that API is running on port 8000 |
| Dashboard says "Disconnected" | API server crashed — check Terminal 1 for errors |
| `CORS error` in browser console | Make sure API runs on port 8000 (not 8001 or other) |

### Training Issues

| Problem | Solution |
|---|---|
| Accuracy stuck at 60% | This is the majority class baseline. Run more rounds, or reduce `noise_multiplier` |
| `NaN` in loss or gradients | Reduce `noise_multiplier` to 0.1, or switch to `fedavg` method |
| Training is very slow | Reduce `num_rounds` or `num_clients`. Use GPU if available |
| "All clients produced NaN" | DP noise too high — reduce `noise_multiplier` in config |
| Per-client accuracy is all identical | Expected for global evaluation — each client evaluates on the same global test set |

### Docker Issues

| Problem | Solution |
|---|---|
| `docker-compose: command not found` | Install Docker Desktop |
| Build fails on `npm ci` | Delete `client/package-lock.json` and rebuild |
| Container exits immediately | Check logs: `docker-compose -f docker/docker-compose.yml logs` |
| Can't access port 3000 | Make sure no other service is using port 3000 |

### Quick Health Check Commands

```bash
# Is the API running?
curl -s http://localhost:8000/api/training/status && echo " ✓ API OK" || echo " ✗ API DOWN"

# Is the dashboard running?
curl -s -o /dev/null -w "%{http_code}" http://localhost:5173 | grep -q 200 && echo "✓ Dashboard OK" || echo "✗ Dashboard DOWN"

# Can we run a quick simulation?
python3 -m pytest tests/test_e2e.py -q && echo "✓ Simulation OK" || echo "✗ Simulation BROKEN"
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────┐
│                    PFL-HCare Quick Reference                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INSTALL:                                                  │
│    pip install -r requirements.txt                         │
│    cd client && npm install && cd ..                       │
│                                                            │
│  VERIFY:                                                   │
│    python3 -m pytest tests/test_e2e.py -v                  │
│                                                            │
│  RUN (CLI only):                                           │
│    python3 scripts/run_local.py \                          │
│        --method pfl_hcare --rounds 30 --clients 5          │
│                                                            │
│  RUN (with Dashboard):                                     │
│    Terminal 1: uvicorn server.main:app --port 8000         │
│    Terminal 2: cd client && npm run dev                     │
│    Terminal 3: python3 scripts/run_local.py ...             │
│    Browser:   http://localhost:5173                         │
│                                                            │
│  RUN (Docker):                                             │
│    docker-compose -f docker/docker-compose.yml up --build  │
│    Browser:   http://localhost:3000                         │
│                                                            │
│  METHODS: fedavg | fedprox | per_fedavg | pfedme |         │
│           pfl_hcare                                        │
│                                                            │
│  DATASETS: har (activity recognition)                      │
│            mimic (medical — auto-falls back to synthetic)  │
│                                                            │
│  STOP:                                                     │
│    Ctrl+C in each terminal, or:                            │
│    pkill -f "uvicorn server.main"                          │
│    pkill -f "vite"                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```
