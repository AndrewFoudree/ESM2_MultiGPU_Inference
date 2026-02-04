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
    &emsp;scripts/<br>
      &emsp;&emsp;benchmark.py<br>
    &emsp;&emsp;Dockerfile<br>
    &emsp;&emsp;requirements.txt<br>
    &emsp;&emsp;requirements-dev.txt<br>
    &emsp;&emsp;pyproject.toml<br>
    &emsp;&emsp;.dockerignore<br>
    &emsp;README.md<br>

   <br> __Scripts:__<br>
   &emsp;benchmark.py<br>
   &emsp;Run Service: unvicorn app.main:app --host 0.0.0.0 --port 8000<br>
   &emsp;Run Benchmark (new terminal): python benchmark.py --url http://localhost:8000 --batch-sizes 1 8 32<br> 

   <br>__Quick Guide__<br>

   &emsp;kubectl apply -f esm2-8gpu-service-yaml<br>
   <br>
   &emsp;Verify the pod is running:<br>
   &emsp;&emsp;kubectl get pods<br>
   &emsp;Access the pod:<br>
   &emsp;kubectl exec -it <pod name> -- /bin/bash<br>
   <br>
   &emsp;Clone the Repostiory:<br>
   &emsp;cd /workspace<br>
   &emsp;git clone <https-repo><br>
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
   &emsp;Start tje Server<br>
   &emsp;uvicorn app.main:app --host 0.0.0.0 --port 8000<br>
   <br>
   &emsp;Verify the Service -  Health Check<br>
   &emsp;curl http://localhost:8000/health | python3 -m json.tool<br>

   &emsp;# Batch prediction (8 sequences = 1 per GPU)<br>
   &emsp;curl -X POST http://localhost:8000/predict/batch \<br><br>
   &emsp;&nbsp;&nbsp;-H "Content-Type: application/json" \<br>
   &emsp;&nbsp;&nbsp;-d '{<br>
   &emsp;&nbsp;&nbsp; "sequences": [<br>
   &emsp;&nbsp;&nbsp;   "MKTVRQERLKSIVRILERSKEPVSGAQL",<br>
   &emsp;&nbsp;&nbsp;   "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",<br>
   &emsp;&nbsp;&nbsp;   "MSGSHHHHHHSSGLVPRGSH",<br>
   &emsp;&nbsp;&nbsp;   "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLL",<br>
   &emsp;&nbsp;&nbsp;   "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM",<br>
   &emsp;&nbsp;&nbsp;   "GLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLFK",<br>
   &emsp;&nbsp;&nbsp;   "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVG",<br>
   &emsp;&nbsp;&nbsp;   "DYKDDDDKGSENLYFQSGSHHHHHHSSGLVPRGS"<br>
   &emsp;&nbsp;&nbsp; ],<br>
   &emsp; "include_embeddings": false<br>
  &emsp;}' | python3 -m json.tool<br>
   
    
