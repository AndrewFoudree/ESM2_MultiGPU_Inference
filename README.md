# ESM2_MultiGPU_Inference
MLOps-ready FastAPI inference service for ESM-2 with automatic multi-GPU scaling, Kubernetes manifests, CI/CD, and benchmarks.

Complete Project Structure:

esm2-multi-gpu-service/
  .github/
    workflows/
      ci.yml
    app/
      __init__.py
      main.py
      model_manager.py
      schemas.py
      config.py
    tests/
      __init__.py
      conftest.py
      test_api.py
      test_gpu_distributor.py
      test_model_manager.py
    k8s/
      namespace.yaml
      deployment.yaml
      service.yaml    
      configmap.yaml
      hpa.yaml
      ingress.yaml
    scripts/
      benchmark.py
    Dockerfile
    requirements.txt
    requirements-dev.txt
    pyproject.toml
    .dockerignore
    README.md
    
