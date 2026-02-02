# ESM2_MultiGPU_Inference<br>
MLOps-ready FastAPI inference service for ESM-2 with automatic multi-GPU scaling, Kubernetes manifests, CI/CD, and benchmarks.
<br>
Complete Project Structure:<br>
<br>
<p>esm2-multi-gpu-service</p>/<br>
  &emsp;.github/<br>
    workflows/<br>
      ci.yml<br>
    app/<br>
      __init__.py<br>
      main.py<br>
      model_manager.py<br>
      schemas.py<br>
      config.py<br>
    tests/<br>
      __init__.py<br>
      conftest.py<br>
      test_api.py<br>
      test_gpu_distributor.py<br>
      test_model_manager.py<br>
    k8s/<br>
      namespace.yaml<br>
      deployment.yaml<br>
      service.yaml<br>    
      configmap.yaml<br>
      hpa.yaml<br>
      ingress.yaml<br>
    scripts/<br>
      benchmark.py<br>
    Dockerfile<br>
    requirements.txt<br>
    requirements-dev.txt<br>
    pyproject.toml<br>
    .dockerignore<br>
    README.md
    
