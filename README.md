
# ESM2_MultiGPU_Inference

[![CI](https://github.com/AndrewFoudree/ESM2_MultiGPU_Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewFoudree/ESM2_MultiGPU_Inference/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MLOps-ready FastAPI inference service for ESM-2 with automatic multi-GPU scaling, Kubernetes manifests, CI/CD, and benchmarks.

---

## 📁 Project Structure
```
esm2-multi-gpu-service/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_manager.py
│   ├── schemas.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_gpu_distributor.py
│   └── test_model_manager.py
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   └── esm2-8gpu-service.yaml    # Used for live testing (adjust namespace as needed)
├── scripts/
│   └── benchmark.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .dockerignore
└── README.md
```

---

## 🚀 Quick Start

### Run the Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

<br>

### Run Benchmark (in a new terminal)
```bash
python scripts/benchmark.py --url http://localhost:8000 --batch-sizes 1 8 32
```

---

## ☸️ Kubernetes Deployment

### Deploy to Kubernetes
```bash
kubectl apply -f k8s/esm2-8gpu-service.yaml
```

<br>

### Verify the pod is running
```bash
kubectl get pods -w
```

<br>

### Access the pod
```bash
kubectl exec -it <pod-name> -- /bin/bash
```

---

## 🛠️ Full Setup Guide

### 1. Clone the Repository
```bash
cd /workspace
git clone https://github.com/AndrewFoudree/ESM2_MultiGPU_Inference.git
cd ESM2_MultiGPU_Inference
```

<br>

### 2. Set up Python Environment
```bash
sudo apt update && sudo apt install -y python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

<br>

### 3. Run Tests
```bash
pytest tests/ -v
```

<br>

### 4. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 API Testing

### Health Check (Should show 8 GPUs)
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

<br>

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL", "include_embeddings": false}'
```

<br>

### Batch Prediction (8 sequences = 1 per GPU)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKTVRQERLKSIVRILERSKEPVSGAQL",
      "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",
      "MSGSHHHHHHSSGLVPRGSH",
      "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLL",
      "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM",
      "GLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLFK",
      "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVG",
      "DYKDDDDKGSENLYFQSGSHHHHHHSSGLVPRGS"
    ],
    "include_embeddings": false
  }' | python3 -m json.tool
```

<br>

### Watch GPU Usage in Real-Time
```bash
watch -n 1 nvidia-smi
```

---

## 🖥️ GPU Architecture

This service is deployed on an AWS `g5.48xlarge` instance with **8 NVIDIA A10G GPUs**.

---

## 🖥️ GPU Architecture

This service is deployed on an AWS `g5.48xlarge` instance with **8 NVIDIA A10G GPUs**.

### How Batch Distribution Works

Sequences are distributed across GPUs using a **contiguous chunking strategy**:

1. Divide batch evenly: `base_per_gpu = batch_size // num_gpus`
2. Calculate remainder: `remainder = batch_size % num_gpus`
3. First `remainder` GPUs each receive one extra sequence
4. All GPUs process their chunks in parallel

**Example: 20 sequences across 8 GPUs**

| GPU | Sequences      | Count |
|-----|----------------|-------|
| 0   | 0, 1, 2        | 3     |
| 1   | 3, 4, 5        | 3     |
| 2   | 6, 7, 8        | 3     |
| 3   | 9, 10, 11      | 3     |
| 4   | 12, 13         | 2     |
| 5   | 14, 15         | 2     |
| 6   | 16, 17         | 2     |
| 7   | 18, 19         | 2     |

This approach maximizes throughput by processing all chunks in parallel using a thread pool, then reordering results to match the original input sequence.

## 🐳 Docker Setup

### 1. Install Docker
```bash
sudo apt update
sudo apt install -y docker.io
```

<br>

### 2. Start Docker
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

<br>

### 3. Add yourself to docker group (avoids needing sudo)
```bash
sudo usermod -aG docker $USER
newgrp docker
```

<br>

### 4. Build the Docker Image
```bash
docker build -t esm2-multi-gpu-inference:latest .
```

<br>

### 5. Run the Container
```bash
# With GPUs
docker run --gpus all -p 8000:8000 esm2-multi-gpu-inference:latest

# Without GPUs (CPU mode)
docker run -p 8000:8000 esm2-multi-gpu-inference:latest
```

---

## 🔧 Linting Setup (Flake8)

This project uses `flake8` for linting with `pre-commit` hooks to ensure code quality on every commit.

### 1. Create the `.flake8` configuration file

Create a `.flake8` file in the project root:
```ini
[flake8]
# Match black's line length
max-line-length = 88

# Match the project's style settings
extend-ignore = 
    # E501: line too long (handled by black)
    E501,
    # W503: line break before binary operator (black preference)
    W503,
    # E203: whitespace before ':' (black compatibility)
    E203

# Exclude directories
exclude = 
    .git,
    __pycache__,
    .venv,
    venv,
    build,
    dist,
    *.egg-info

# Per-file ignores
per-file-ignores = 
    # Allow assert in tests
    tests/*: S101

# Show source code for errors
show-source = true

# Count errors
count = true
```

<br>

### 2. Install pre-commit and flake8
```bash
pip install pre-commit flake8
```

<br>

### 3. Create `.pre-commit-config.yaml`

Create a `.pre-commit-config.yaml` file in the project root:
```yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-flake8
    rev: v7.0.0
    hooks:
      - id: flake8
        args: [--config=.flake8]

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
```

<br>

### 4. Install the pre-commit hooks
```bash
pre-commit install
```

<br>

### 5. Run linting manually
```bash
# Run flake8 on all files
pre-commit run --all-files flake8

# Run all pre-commit hooks
pre-commit run --all-files

# Or run flake8 directly
flake8 app/ tests/ scripts/
```

<br>

### 6. Automatic linting on commit

Once installed, pre-commit will automatically run flake8 (and other hooks) every time you run `git commit`. If any checks fail, the commit will be blocked until you fix the issues.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
---
