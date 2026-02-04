# ESM2_MultiGPU_Inference<br>
MLOps-ready FastAPI inference service for ESM-2 with automatic multi-GPU scaling, Kubernetes manifests, CI/CD, and benchmarks.
<br>
<br>__Complete Project Structure:__<br>
<br>
esm2-multi-gpu-service/<br>
  &emsp;.github/<br>
    &emsp;&emsp;workflows/<br>
      &emsp;&emsp;&emsp;ci.yml<br>
    &emsp;app/<br>
      &emsp;&emsp;\_\_init\_\_.py<br>
      &emsp;&emsp;main.py<br>
      &emsp;&emsp;model_manager.py<br>
      &emsp;&emsp;schemas.py<br>
      &emsp;&emsp;config.py<br>
    &emsp;tests/<br>
      &emsp;&emsp;\_\_init\_\_.py<br>
      &emsp;&emsp;conftest.py<br>
      &emsp;&emsp;test_api.py<br>
      &emsp;&emsp;test_gpu_distributor.py<br>
      &emsp;&emsp;test_model_manager.py<br>
    &emsp;k8s/<br>
      &emsp;&emsp;namespace.yaml<br>
      &emsp;&emsp;deployment.yaml<br>
      &emsp;&emsp;service.yaml<br>
      &emsp;&emsp;configmap.yaml<br>
      &emsp;&emsp;hpa.yaml<br>
      &emsp;&emsp;ingress.yaml<br>
      &emsp;&emsp;esm2-8gpu-service.yaml **\--used for live testing**<br>
      &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**\--adjust namespace as needed**<br>
    &emsp;scripts/<br>
      &emsp;&emsp;benchmark.py<br>
    &emsp;Dockerfile<br>
    &emsp;requirements.txt<br>
    &emsp;requirements-dev.txt<br>
    &emsp;pyproject.toml<br>
    &emsp;.dockerignore<br>
    &emsp;README.md<br>

   # Scripts:<br>
   &emsp;benchmark.py<br>
   &emsp;Run Service: unvicorn app.main:app --host 0.0.0.0 --port 8000<br>
   &emsp;Run Benchmark (new terminal): python benchmark.py --url http://localhost:8000 --batch-sizes 1 8 32<br> 

   # ESM2 Multi-GPU Inference Service - Quick Guide<br>

   &emsp;kubectl apply -f esm2-8gpu-service-yaml<br>
   <br>
   &emsp;Verify the pod is running:<br>
   &emsp;&emsp;kubectl get pods<br><br>
   &emsp;Access the pod:<br>
   &emsp;kubectl exec -it insert-pod-name -- /bin/bash<br>
   <br>
   &emsp;Clone the Repository:<br>
   &emsp;cd /workspace<br>
   &emsp;git clone insert-https-repo-code-here<br>
   &emsp;cd ESM2_MultiGPU_Inference<br>
   <br>
   &emsp;Set up Python Environment:<br>
   &emsp;apt update && apt install -y python3.10-venv<br>
   &emsp;python3 -m venv .venv<br>
   &emsp;source .venv/bin/activate<br>
   &emsp;pip install requirements<br>
   <br>
   &emsp;Run Tests<br>
   &emsp;pytest tests/ -v<br>
   <br>
   &emsp;Start the Server<br>
   &emsp;uvicorn app.main:app --host 0.0.0.0 --port 8000<br>
   <br>

# Health Check - Should Show 8 GPUs
curl http://localhost:8000/health | python3 -m json.tool
# Single Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL", "include_embeddings": false}'
# Batch Prediction (8 sequences = 1 per GPU)
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
    <br>
    
   # Watch GPU usage in real-time:<br>
   &emsp;watch -n 1 nvidia-smi<br>
   # Docker:<br>
   &emsp;# Install Docker<br>
   &emsp;sudo apt update<br>
   &emsp;sudo apt install -y docker.io<br><br>
   &emsp;# Start Docker<br>
   &emsp;sudo systemctl start docker<br>
   &emsp;sudo systemctl enable docker<br><br>
   &emsp;# Add yourself to docker group (avoids needing sudo)<br>
   &emsp;&emsp;sudo usermod -aG docker $USER<br>
   &emsp;&emsp;newgrp docker<br><br>
    &emsp;# Build the Docker Image<br>
    &emsp;&emsp;&emsp;docker build -t esm2-multi-gpu-inference:latest .<br><br>
