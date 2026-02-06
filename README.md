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
│   └── esm2-8gpu-service.yaml
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

### Run Benchmark (in a new terminal)
```bash
python scripts/benchmark.py --url http://localhost:8000 --batch-sizes 1 8 32
```

---

## 🖥️ GPU Architecture

This service is designed for AWS `g5.48xlarge` instances with **8 NVIDIA A10G GPUs**.

### Auto-Detection

The service automatically detects available GPUs at startup:
- **0 GPUs**: Falls back to CPU inference
- **1 GPU**: Single-GPU mode
- **4 GPUs**: Distributes across 4 GPUs
- **8 GPUs**: Full multi-GPU distribution

### GPU Distribution Strategy

Sequences are distributed across GPUs using a **contiguous chunking strategy**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Incoming Batch (20 sequences)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GPUDistributor                             │
│  base_per_gpu = 20 // 8 = 2                                     │
│  remainder = 20 % 8 = 4                                         │
│  First 4 GPUs get 3 sequences, remaining 4 GPUs get 2           │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   GPU 0     │     │   GPU 1     │     │   GPU 2-7   │
   │  Seq 0,1,2  │     │  Seq 3,4,5  │     │    ...      │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Results Gathered & Reordered                       │
└─────────────────────────────────────────────────────────────────┘
```

**Distribution Example: 20 sequences across 8 GPUs**

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

### Batch Distribution Pseudocode

```python
def distribute_batch(sequences, gpu_count):
    """
    Distribute sequences across GPUs using contiguous chunking.
    
    Algorithm:
    1. Calculate base sequences per GPU (integer division)
    2. Calculate remainder sequences
    3. First 'remainder' GPUs get one extra sequence
    4. Assign contiguous chunks to each GPU
    """
    num_sequences = len(sequences)
    
    # Handle no GPUs - run on CPU
    if gpu_count == 0:
        return [(0, all_indices, sequences)]
    
    distribution = []
    current_idx = 0
    
    # Even distribution with remainder handling
    base_per_gpu = num_sequences // gpu_count
    remainder = num_sequences % gpu_count
    
    for gpu_id in range(gpu_count):
        # First 'remainder' GPUs get one extra
        count = base_per_gpu + (1 if gpu_id < remainder else 0)
        
        if count > 0:
            indices = range(current_idx, current_idx + count)
            gpu_sequences = sequences[current_idx:current_idx + count]
            distribution.append((gpu_id, indices, gpu_sequences))
            current_idx += count
    
    return distribution
```

### Multi-GPU Inference Flow

```python
async def predict_batch(sequences):
    """
    Multi-GPU batch inference with parallel execution.
    
    Flow:
    1. Validate all sequences
    2. Get distribution plan from GPUDistributor
    3. Submit inference tasks to ThreadPoolExecutor (one per GPU)
    4. Await all results in parallel
    5. Reorder results to match original sequence order
    6. Return combined results with distribution metadata
    """
    # Get distribution plan
    distribution = gpu_distributor.distribute_batch(sequences)
    
    # Submit parallel tasks
    tasks = []
    for gpu_id, indices, gpu_sequences in distribution:
        task = executor.submit(run_inference, gpu_id, gpu_sequences)
        tasks.append((indices, task))
    
    # Gather and reorder results
    all_results = [None] * len(sequences)
    for indices, task in tasks:
        results = await task
        for idx, result in zip(indices, results):
            all_results[idx] = result
    
    return all_results
```

---

## ⚡ Batch Queue (Request Batching)

For high-throughput scenarios with many concurrent single-sequence requests, the service includes an optional **batch queue** that automatically combines requests.

### How It Works

1. Concurrent `/predict` requests are collected into a queue
2. The queue waits up to 50ms for additional requests
3. All collected requests are processed as a single batch
4. Individual results are returned to each caller

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `BATCH_QUEUE_ENABLED` | `true` | Enable/disable request batching |
| `BATCH_QUEUE_MAX_WAIT_MS` | `50` | Max time to wait for more requests |
| `BATCH_QUEUE_MAX_SIZE` | `64` | Max requests to batch together |

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Low traffic, latency-sensitive | Disable batch queue |
| Bursty traffic, throughput matters | Enable with 20-50ms wait |
| High concurrency | Enable with larger batch size |

---

## 📊 Expected Performance Characteristics

### Throughput Scaling by GPU Count

| GPUs | Relative Throughput | Notes |
|------|---------------------|-------|
| 1    | 1x (baseline)       | Single GPU processes all sequences |
| 4    | ~3.5-3.8x           | Near-linear scaling |
| 8    | ~6.5-7.5x           | Good scaling with some overhead |

### Factors Affecting Performance

- **Sequence length**: Longer sequences = more compute per sequence
- **Batch size**: Larger batches amortize overhead better
- **GPU memory**: A10G has 24GB; model uses ~2-3GB per replica
- **Inter-GPU overhead**: ThreadPool coordination adds minimal latency

### Benchmark Results (Expected)

```
Batch Size | 1 GPU    | 4 GPUs   | 8 GPUs
-----------|----------|----------|----------
1          | 50ms     | 50ms     | 50ms
8          | 400ms    | 120ms    | 70ms
32         | 1600ms   | 420ms    | 230ms
64         | 3200ms   | 820ms    | 430ms
```

*Note: Actual results vary based on sequence length and hardware.*

---

## ☸️ Kubernetes Deployment

### Production Deployment
```bash
# Create namespace
kubectl create namespace esm2-inference

# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# Deploy service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Development/Testing Pod
```bash
kubectl apply -f k8s/esm2-8gpu-service.yaml
kubectl exec -it esm2-8gpu -n hackathon -- /bin/bash
```

### Verify Deployment
```bash
kubectl get pods -n esm2-inference -w
kubectl logs -f deployment/esm2-inference -n esm2-inference
```

---

## 🛠️ Full Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/AndrewFoudree/ESM2_MultiGPU_Inference.git
cd ESM2_MultiGPU_Inference
```

### 2. Set up Python Environment
```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 API Testing

### Health Check
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL", "include_embeddings": false}'
```

### Batch Prediction (8 sequences)
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

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## 🐳 Docker Setup

### Build the Image
```bash
docker build -t esm2-multi-gpu-inference:latest .
```

### Run the Container
```bash
# With GPUs
docker run --gpus all -p 8000:8000 esm2-multi-gpu-inference:latest

# Without GPUs (CPU mode)
docker run -p 8000:8000 esm2-multi-gpu-inference:latest
```

---

## 🔧 Development

### Linting
```bash
# Run all linters
pre-commit run --all-files

# Run flake8 only
flake8 app/ tests/ scripts/
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 🚧 Future Improvements

Given more time, I would implement the following enhancements:

### Performance
- **Dynamic batching with padding optimization**: Group sequences by similar length to reduce padding overhead
- **CUDA streams**: Use multiple CUDA streams per GPU for overlapping compute and memory transfers
- **Model quantization**: INT8 or FP16 quantization for faster inference with minimal accuracy loss
- **TensorRT optimization**: Convert model to TensorRT for optimized GPU inference

### Scalability
- **Horizontal pod autoscaling**: Scale replicas based on request queue depth
- **Model sharding**: For larger ESM-2 variants (3B parameters), shard across GPUs
- **Redis-based request queue**: External queue for better load distribution across pods
- **gRPC endpoint**: Lower latency than REST for high-frequency clients

### Observability
- **Prometheus metrics**: GPU utilization, inference latency histograms, queue depth
- **Distributed tracing**: OpenTelemetry integration for request tracing
- **Alerting**: PagerDuty/Slack alerts for GPU errors or latency spikes

### Reliability
- **Circuit breaker**: Fail fast when GPUs are overloaded
- **Request prioritization**: Priority queues for different client tiers
- **Graceful degradation**: Automatic fallback to fewer GPUs if some fail
- **Chaos testing**: GPU failure injection for resilience testing

### Developer Experience
- **Helm chart**: Parameterized Kubernetes deployment
- **Terraform modules**: Infrastructure as code for AWS resources
- **Load testing suite**: Locust or k6 scripts for performance regression testing

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
