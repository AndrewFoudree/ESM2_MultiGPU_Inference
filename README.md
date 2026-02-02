# ESM2_MultiGPU_Inference<br>
MLOps-ready FastAPI inference service for ESM-2 with automatic multi-GPU scaling, Kubernetes manifests, CI/CD, and benchmarks.
<br>
Complete Project Structure:<br>
<br>
esm2-multi-gpu-service/
  &emsp;.github/<br>
    &emsp;&emsp;workflows/<br>
      &emsp;&emsp;&emsp;ci.yml<br>
    &emsp;app/<br>
      &emsp;&emsp;__init__.py<br>
      &emsp;&emsp;main.py<br>
      &emsp;&emsp;model_manager.py<br>
      &emsp;&emsp;schemas.py<br>
      &emsp;&emsp;config.py<br>
    &emsp;tests/<br>
      &emsp;&emsp;__init__.py<br>
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
    &emsp;README.md
    
