<div align="center">

&nbsp;

# 🏥 PFL-HCare
### Personalized Federated Learning for Privacy-Preserving and Scalable IoT-Driven Smart Healthcare

**TTEH LAB · School of Engineering, Dayananda Sagar University**

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower FL](https://img.shields.io/badge/Federated_Learning-Flower_1.5-f5a623?style=for-the-badge)](https://flower.ai/)
[![React](https://img.shields.io/badge/Dashboard-React_18-61dafb?style=for-the-badge&logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![IEEE](https://img.shields.io/badge/IEEE-ICICI_2025-00629b?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

&nbsp;

*Prototype implementation of:*

**"Personalized Federated Learning for Privacy-Preserving and Scalable IoT-Driven Smart Healthcare"**

*ICICI-2025, IEEE Xplore · DOI: [10.1109/ICICI65870.2025.11069877](https://doi.org/10.1109/ICICI65870.2025.11069877)*

&nbsp;

</div>

---

## 🔭 Overview

The exponential growth of IoT in healthcare has transformed patient monitoring and diagnosis, but traditional centralized machine learning methods present critical challenges in data privacy, scalability, and adaptability to diverse patient conditions. This work presents **PFL-HCare**, a Personalized Federated Learning framework for IoT-driven smart healthcare that enforces _"train locally, share globally"_ through four integrated components: a **MAML-based meta-learning personalizer** for adaptive model customization, a **Differential Privacy mechanism** with RDP accounting for formal privacy guarantees, **k-bit gradient quantization** for communication-efficient updates, and **adaptive client selection** based on gradient norms for optimal convergence. All components are trained via Federated Learning — ensuring that sensitive patient data never leaves the edge device. The framework includes a real-time **React + FastAPI dashboard** for live visualization of convergence, privacy budget, communication overhead, and per-client metrics across five FL methods. Experimental evaluations on UCI HAR and synthetic medical datasets demonstrate up to **89.1% accuracy** on FedAvg baseline and competitive performance with differential privacy enabled, while achieving **75% bandwidth savings** through gradient quantization.

`Personalized Federated Learning` &nbsp;·&nbsp; `MAML Meta-Learning` &nbsp;·&nbsp; `Differential Privacy` &nbsp;·&nbsp; `IoT Healthcare` &nbsp;·&nbsp; `Gradient Quantization` &nbsp;·&nbsp; `Real-Time Dashboard`

---

## 📋 Table of Contents

1. [Problem Statement](#1--problem-statement)
2. [Proposed Architecture](#2--proposed-architecture)
3. [How It Works](#3--how-it-works)
4. [Paper Results & Metrics](#4--paper-results--metrics)
5. [Code Architecture](#5--code-architecture)
6. [Core Modules — Deep Dive](#6--core-modules--deep-dive)
7. [Setup & Usage](#7--setup--usage)
8. [Implementation Results](#8--implementation-results)
9. [Implementation Limitations](#9--implementation-limitations)

---

## 1. 🔍 Problem Statement

> *"Can we train accurate healthcare AI without ever seeing the patient data?"*

Traditional centralized machine learning in healthcare requires aggregating sensitive patient data — medical records, wearable sensor readings, diagnostic images — into a single location. This creates severe **privacy risks** (HIPAA/GDPR violations, data breach targets), **scalability bottlenecks** (bandwidth costs of transmitting raw medical data), and **personalization failures** (a one-size-fits-all global model cannot adapt to individual patient conditions, demographics, and sensor variations).

Standard Federated Learning (FL) addresses privacy by keeping data on-device, but introduces new challenges:

- **Heterogeneous Data Distributions** — Medical data is fundamentally non-IID across patients due to varied demographics, disease states, and sensor hardware
- **Personalization Gap** — A global FL model may not generalize to individual patients, reducing predictive performance by 10-20%
- **Privacy Leakage** — Gradient updates exchanged during FL can be exploited through model inversion and membership inference attacks
- **Communication Overhead** — Repeated client-server communication strains bandwidth-constrained IoT edge devices

**What's needed →** A federated learning framework that provides **personalized models** for each patient/device while maintaining **formal privacy guarantees**, **communication efficiency**, and **scalability** across hundreds of IoT healthcare nodes.

---

## 2. 🏗️ Proposed Architecture

PFL-HCare implements adaptive personalized federated learning through **four tightly integrated modules** that work in concert to balance personalization, privacy, and efficiency.

| # | Module | Role | Key Output |
|---|---|---|---|
| 1️⃣ | **MAML Personalizer** | Meta-learning-based local model adaptation | Personalized client weights w_i |
| 2️⃣ | **DP Mechanism** | Gaussian noise injection with RDP accounting | (ε, δ)-DP guaranteed updates |
| 3️⃣ | **Gradient Quantizer** | k-bit compression of model updates (Eq.8) | 75% bandwidth reduction |
| 4️⃣ | **Adaptive Selector** | Gradient-norm-based client selection (Eq.9) | Optimized participation |
| 5️⃣ | **Secure Aggregator** | Simulated homomorphic encryption workflow | Encrypted aggregation pipeline |

### 🖥️ Full-Stack Dashboard

The framework includes a complete web dashboard for real-time visualization:

```
┌────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend ──WebSocket──► React Dashboard (5 Views)          │
│                                                                    │
│  📊 Overview    — KPI cards + network topology + activity feed     │
│  📈 Convergence — Multi-line accuracy chart for all 5 methods      │
│  🔒 Privacy     — Epsilon gauge + privacy-accuracy tradeoff        │
│  📡 Communication — Bandwidth bars + client participation heatmap  │
│  🏆 Comparison  — Live tables reproducing paper Tables II-IV       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. ⚡ How It Works

### 🔄 Federated Training — Privacy by Design

Each FL round follows a strict privacy-preserving cycle:

```
📡 Server broadcasts global model weights w*
   ↓
🏋️ Each client performs personalized local training (MAML inner loop)
   ↓
✂️ Gradient clipping (L2 norm bound) + DP Gaussian noise injection (Eq.5)
   ↓
📦 k-bit gradient quantization compresses updates (Eq.8)
   ↓
🔒 Simulated homomorphic encryption of model updates (Eq.6)
   ↓
📤 Compressed, noisy update sent to server
   ↓
⚖️ Server aggregates via weighted FedAvg (Eq.1), scaled by dataset size
   ↓
🎯 Adaptive client selection for next round based on gradient norms (Eq.9)
```

### 📐 Key Equations

**Global FL Objective (Eq.1):**
```
min_w Σᵢ (|Dᵢ| / Σⱼ|Dⱼ|) × Fᵢ(w)
```

**MAML Meta-Initialization (Eq.3):**
```
w* = argmin_w Σᵢ Fᵢ(w - α∇Fᵢ(w))
```

**Client Fine-Tuning (Eq.4):**
```
wᵢ = w* - α∇Fᵢ(w*)
```

**Differential Privacy Noise (Eq.5):**
```
wᵢ' = wᵢ + N(0, σ²)    guarantees (ε, δ)-DP
```

**Gradient Quantization (Eq.8):**
```
Q(wᵢ) = round((wᵢ - w_min) / (w_max - w_min) × (2ᵏ - 1))
```

**Adaptive Client Selection (Eq.9):**
```
pᵢ = ‖∇Fᵢ(w)‖ / Σⱼ‖∇Fⱼ(w)‖
```

### 🏆 Methods Compared

All five methods share the same model architecture, data partitions, and evaluation protocol:

| Method | Personalized | Diff. Privacy | Secure Agg. | Quantization | Adaptive Sel. |
|---|:---:|:---:|:---:|:---:|:---:|
| FedAvg | ✗ | ✗ | ✗ | ✗ | ✗ |
| FedProx | ~ | ✗ | ✗ | ✗ | ✗ |
| Per-FedAvg | ✓ | ~ | ✗ | ✗ | ✗ |
| pFedMe | ✓ | ✓ | ✗ | ✗ | ✗ |
| **PFL-HCare (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 4. 📊 Paper Results & Metrics

> 📝 All values from paper Section IV. Implementation results on scaled-down prototype below in Section 8.

### 🎯 Model Convergence and Accuracy (Table II)

| Model | Test Accuracy (MIMIC-III) | Test Accuracy (UCI HAR) | Convergence Speed |
|---|:---:|:---:|:---:|
| FedAvg | 87.50% | 89.30% | Baseline |
| FedProx | 89.20% | 91.00% | +8.5% Faster |
| Per-FedAvg | 90.50% | 92.50% | +15.2% Faster |
| pFedMe | 91.60% | 93.20% | +21.3% Faster |
| **🏆 PFL-HCare (Ours)** | **92.30%** | **94.10%** | **+27.8% Faster** |

### 🔒 Privacy-Preserving Effectiveness (Table III)

| Model | DP Applied | Accuracy Drop | Privacy Parameters |
|---|:---:|:---:|:---:|
| FedAvg | ✗ No DP | N/A | N/A |
| FedProx | ✗ No DP | N/A | N/A |
| Per-FedAvg | ✓ Partial | 3.10% | ε=3.5, δ=10⁻⁴ |
| pFedMe | ✓ DP Applied | 2.50% | ε=2.8, δ=10⁻⁵ |
| **🏆 PFL-HCare (Ours)** | **✓ DP + Secure Agg.** | **1.70%** | **ε=2.1, δ=10⁻⁵** |

### 📡 Communication Overhead and Scalability (Table IV)

| Metric | FedAvg | FedProx | pFedMe | **PFL-HCare** |
|---|:---:|:---:|:---:|:---:|
| Communication Overhead | High | High | Medium | **Low (-38.2%)** |
| Bandwidth Consumption | High | High | Medium | **Low (-45%)** |
| Client Participation | Random | Fixed | Adaptive | **Optimized (+40%)** |
| Scalability (N=500) | 15% slowdown | 10% slowdown | 5% slowdown | **Stable** |

---

## 5. 🗂️ Code Architecture

The prototype translates the paper's architecture into a layered modular Python/TypeScript package. Each layer is independently testable and deployable.

```
Personalized_Federated_Learning/
├── pfl_hcare/                          # 🧠 Core ML library
│   ├── models/
│   │   ├── health_classifier.py        # 🏥 MLP for medical prediction (~15K params)
│   │   └── har_classifier.py           # 📱 1D-CNN for activity recognition (~52K params)
│   ├── fl/
│   │   ├── server.py                   # 🖥️ FL simulation engine (local, no Ray)
│   │   ├── client.py                   # 📡 Flower NumPyClient — 5 strategy modes
│   │   ├── aggregation.py              # ⚖️ Weighted FedAvg (Eq.1)
│   │   └── strategies/                 # 🎯 One strategy class per method
│   │       ├── fedavg.py               #     Vanilla weighted averaging
│   │       ├── fedprox.py              #     + proximal regularization (μ=0.01)
│   │       ├── per_fedavg.py           #     + MAML personalization
│   │       ├── pfedme.py               #     + Moreau envelope (λ=15)
│   │       └── pfl_hcare.py            #     + DP + quantization + adaptive selection
│   ├── maml/
│   │   └── maml.py                     # 🔁 MAML inner/outer loop (FOMAML toggle)
│   ├── privacy/
│   │   ├── differential_privacy.py     # 🔐 Gaussian DP with RDP accounting (Eq.5)
│   │   ├── secure_aggregation.py       # 🔒 Simulated HE for dashboard (Eq.6-7)
│   │   └── quantization.py             # 📦 k-bit gradient compression (Eq.8)
│   └── metrics/
│       └── collector.py                # 📊 Per-round metric tracking + callbacks
│
├── data/                               # 📂 Dataset layer
│   ├── har_loader.py                   # 📱 UCI HAR — 10,299 samples, 6 activities
│   ├── mimic_loader.py                 # 🏥 4-tier fallback: MIMIC-III → Demo → Heart → Synthetic
│   ├── synthetic_generator.py          # 🤖 Tier 4: configurable vital signs generator
│   └── partition.py                    # 🔀 Dirichlet non-IID partitioning
│
├── server/                             # 🌐 FastAPI backend
│   ├── main.py                         # 🚀 App entry point + CORS + lifespan
│   ├── db.py                           # 💾 SQLite persistence (runs + round metrics)
│   ├── orchestrator.py                 # 🎼 Sequential comparison run manager
│   ├── routes/
│   │   ├── training.py                 # ▶️ POST /start · POST /stop · GET /status
│   │   ├── metrics.py                  # 📈 GET /runs · GET /{run_id}
│   │   └── datasets.py                 # 📂 GET /info · POST /partition-preview
│   └── ws/
│       └── live.py                     # 📡 WebSocket live metric streaming
│
├── client/                             # 🎨 React dashboard
│   └── src/
│       ├── components/
│       │   ├── layout/                 # Sidebar · Header · ControlRibbon
│       │   ├── views/                  # 5 views: Overview · Convergence · Privacy · Comm · Comparison
│       │   ├── charts/                 # ConvergenceChart · PrivacyGauge · BandwidthChart
│       │   └── widgets/                # KpiCard · ActivityFeed
│       ├── hooks/                      # useWebSocket · useTrainingState
│       └── types/                      # TypeScript interfaces for all metrics
│
├── docker/                             # 🐳 Docker simulation
│   ├── Dockerfile.api                  # FastAPI container
│   ├── Dockerfile.dashboard            # React + nginx container
│   └── docker-compose.yml              # Multi-container orchestration
│
├── configs/
│   ├── default.yaml                    # 🎛️ All hyperparameters (single source of truth)
│   └── comparison.yaml                 # 🏆 Full 5-method comparison config
│
├── scripts/
│   ├── run_local.py                    # 🚀 CLI simulation launcher
│   └── download_data.py                # 📥 Dataset downloader
│
└── tests/                              # 🧪 49 tests across 11 test files
    ├── test_models.py                  # Model forward pass, param counts, gradients
    ├── test_maml.py                    # MAML inner/outer loop, second-order mode
    ├── test_dp.py                      # DP noise, clipping, epsilon tracking
    ├── test_quantization.py            # k-bit quantize/dequantize, bandwidth
    ├── test_secure_agg.py              # Encrypt/decrypt round-trip, aggregation
    ├── test_partition.py               # Dirichlet partitioning, heterogeneity score
    ├── test_strategies.py              # All 5 strategy instantiation
    └── test_e2e.py                     # End-to-end FedAvg + PFL-HCare smoke tests
```

### 🎛️ Key Configuration Parameters

All hyperparameters are centralized in `configs/default.yaml`:

| Parameter | Value | Notes |
|---|:---:|---|
| `learning_rate` | 0.01 | SGD learning rate for local training |
| `batch_size` | 32 | Per-client batch size |
| `num_clients` | 10 | Number of federated IoT clients |
| `num_rounds` | 200 | Communication rounds |
| `local_epochs` | 5 | Local training epochs per round |
| `noise_multiplier` σ | 0.5 | DP noise scale (privacy-accuracy tradeoff) |
| `max_grad_norm` | 1.0 | Gradient clipping bound |
| `k_bits` | 8 | Quantization bit-width (2, 4, 8, 16) |
| `partition_alpha` | 0.5 | Dirichlet non-IID concentration |
| `inner_lr` | 0.01 | MAML inner loop learning rate |
| `inner_steps` | 5 | MAML inner loop gradient steps |

---

## 6. 🧩 Core Modules — Deep Dive

### 🏥 Health Classifier — MLP for Medical Prediction
> 📁 `pfl_hcare/models/health_classifier.py`

3-layer MLP processing 13-feature medical vital signs (heart rate, blood pressure, SpO2, temperature, etc.). Designed for IoT edge deployment with only ~15K parameters (61 KB):

```
Input (13 features) → Dense(128)+BN+ReLU+Dropout(0.3)
                     → Dense(64)+BN+ReLU+Dropout(0.2)
                     → Dense(32)+ReLU
                     → Output (2 classes: healthy/at-risk)
```

### 📱 HAR Classifier — 1D-CNN for Activity Recognition
> 📁 `pfl_hcare/models/har_classifier.py`

1D-CNN processing 9-channel accelerometer/gyroscope signals from wearable IoT devices. ~52K parameters (209 KB) — feasible for Raspberry Pi deployment:

```
Input (9ch × 128 timesteps) → Conv1D(64,k=5)+BN+ReLU+MaxPool
                             → Conv1D(128,k=3)+BN+ReLU+MaxPool
                             → Conv1D(64,k=3)+BN+ReLU
                             → GlobalAvgPool → Dense(64)+Dropout(0.3)
                             → Output (6 activities)
```

### 🔁 MAML — Meta-Learning Personalizer
> 📁 `pfl_hcare/maml/maml.py` · Implements Eqs. 3–4

Model-Agnostic Meta-Learning that learns an adaptable global initialization w* such that each client can rapidly fine-tune to patient-specific data. Supports both full second-order MAML (Hessian-vector products) and FOMAML (first-order approximation) for resource-constrained devices.

### 🔐 Differential Privacy — Formal Privacy Guarantees
> 📁 `pfl_hcare/privacy/differential_privacy.py` · Implements Eq. 5

Gaussian mechanism with per-sample gradient clipping and RDP (Renyi Differential Privacy) accounting. Configurable noise multiplier σ controls the privacy-accuracy tradeoff — the dashboard visualizes this in real-time with an interactive σ slider.

```python
# Per-round privacy pipeline
clipped = dp.clip_gradients(model_updates)      # L2 norm bound
noisy = dp.add_noise(clipped, sample_rate=q)     # N(0, σ²) injection
epsilon = dp.get_epsilon()                        # Cumulative ε tracking
```

### 📦 Gradient Quantization — Communication Efficiency
> 📁 `pfl_hcare/privacy/quantization.py` · Implements Eq. 8

k-bit encoding compresses float32 model weights to k-bit integers before transmission, reducing bandwidth by up to 75% (8-bit) or 93.75% (2-bit). Dequantization on server before aggregation.

### 🎯 Adaptive Client Selection — Optimized Participation
> 📁 `pfl_hcare/fl/strategies/pfl_hcare.py` · Implements Eq. 9

After round 1, the server tracks gradient norms per client. Selection probability is proportional to gradient magnitude — clients with more to learn participate more frequently, reducing wasted communication for already-converged clients.

### 📡 FL Strategies — Five Methods Under One Roof
> 📁 `pfl_hcare/fl/strategies/`

All five methods share the same Flower client infrastructure with swapped strategy logic:

| Strategy | Key Mechanism | Complexity |
|---|---|---|
| **FedAvg** | Weighted parameter averaging | Baseline |
| **FedProx** | + proximal term `(μ/2)‖w - w_global‖²` | + 1 regularizer |
| **Per-FedAvg** | + MAML inner/outer loop | + meta-learning |
| **pFedMe** | + Moreau envelope `(λ/2)‖w - θᵢ‖²` | + personal params |
| **PFL-HCare** | + MAML + DP + quantization + adaptive selection | Full pipeline |

---

## 7. 🚀 Setup & Usage

> 📖 For detailed step-by-step instructions, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**

### ⚙️ Hardware Requirements

| Component | This Prototype | Paper Reproduction |
|---|---|---|
| 🎮 GPU | Optional (CPU works) | NVIDIA A100 · 80 GB VRAM |
| 🧠 RAM | 4 GB+ | 16 GB+ |
| 💾 Disk | ~2 GB (deps + data) | ~5 GB |
| 📡 FL Clients | 5–10 | 100 |
| 🔄 FL Rounds | 30–50 | 200 |

### 📦 Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd client && npm install && cd ..

# Start API + Dashboard
uvicorn server.main:app --port 8000 &
cd client && npm run dev &

# Run FL simulation
python3 scripts/run_local.py --method pfl_hcare --rounds 30 --clients 5 --dataset mimic
```

Open **http://localhost:5173** to view the live dashboard.

### 📂 Datasets

| Dataset | Purpose | Size | Access |
|---|---|---|---|
| **UCI HAR** | Activity recognition (wearable IoT) | 10,299 samples · 561 features · 6 classes | Free auto-download |
| **MIMIC-III** | ICU mortality prediction | 48,520 stays · 17 features | PhysioNet credentials |
| **Heart Disease UCI** | Cardiac risk classification | 303 samples · 13 features | Free auto-download |
| **Synthetic Medical** | Fallback vital signs data | Configurable · 13 features · 3 clusters | Always available |

---

## 8. 📊 Implementation Results

The following results are from running the prototype on consumer-grade hardware (Apple M-series, CPU) with scaled-down parameters (5 clients, 30 rounds).

### 📈 FedAvg Convergence

```
Round  1 — accuracy: 88.4%   loss: 0.4515
Round  5 — accuracy: 87.3%   loss: 0.3754
Round 10 — accuracy: 88.0%   loss: 0.4148
Round 20 — accuracy: 88.9%   loss: 0.4095
Round 30 — accuracy: 88.8%   loss: 0.4045   ← stable convergence
```

### 🔒 PFL-HCare with Differential Privacy

```
Round  1 — accuracy: 60.0%   loss: 84.04   (cold start with DP noise)
Round  5 — accuracy: 72.7%   loss: 54.58   (learning despite noise)
Round 13 — accuracy: 81.9%   loss:  2.56   (convergence through noise)
Round 24 — accuracy: 87.6%   loss:  3.48   ← peak with DP enabled
```

The variance in PFL-HCare is **expected** — it directly demonstrates the privacy-accuracy tradeoff that the dashboard's Privacy Panel is designed to visualize. Higher σ = more privacy + more variance.

### 🧪 Test Suite

```
49 passed in 1.73s (excluding HAR download tests)
```

---

## 9. ⚠️ Implementation Limitations

| # | 📄 Paper Spec | 💻 Prototype Reality | 🔧 Path to Fix |
|---|---|---|---|
| L1 | 100 FL clients on A100 GPU | 5–10 clients on CPU | Scale `num_clients` with GPU hardware |
| L2 | Real MIMIC-III (48K ICU stays) | Synthetic fallback (2K samples) | Get PhysioNet credentials → swap data path |
| L3 | Homomorphic encryption (Paillier) | Simulated with latency + status tracking | Replace with TenSEAL or Paillier library |
| L4 | Full MAML second-order gradients | Personalized training with regularizer | Enable `second_order=True` for small models |
| L5 | 200 FL rounds | 30 rounds (time constraint) | Increase `num_rounds` in config |
| L6 | D3 network topology visualization | Placeholder in Overview panel | Implement force-directed graph |
| L7 | PDF/LaTeX export from dashboard | Not implemented | Add reportlab-based export_report.py |
| L8 | UCI HAR 9-channel CNN input | 561-flat feature input (auto-reshaped) | Use raw inertial signal files |

---

<div align="center">

## 👥 Authors

Dilip Kumar Jang Bahadur Saini · Nilesh Shelke · Amit Pimpalkar · Prajwalasimha S N · Ranjima P · Vinitha V

**Department of Computer Science and Engineering (Cyber Security)**
School of Engineering, Dayananda Sagar University

---

**TTEH LAB · School of Engineering · Dayananda Sagar University**

*Bangalore – 562112, Karnataka, India*

&nbsp;

[![IEEE Paper](https://img.shields.io/badge/Read_the_Paper-IEEE_Xplore-00629b?style=flat-square&logo=ieee)](https://doi.org/10.1109/ICICI65870.2025.11069877)

</div>
