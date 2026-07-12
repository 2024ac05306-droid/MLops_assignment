# MLops_assignment
This repository is  created for Mlops assignment. 

**Purpose:** An MLOps Experimental Learning Assignment focused on end-to-end ML model development with emphasis on:

1. ML model design and development
2. CI/CD pipelines and automation
3. Containerization and deployment
4. Experiment tracking
5. Data versioning and reproducibility (DVC)
6. Cloud deployment and monitoring
7. Production-ready MLOps best practices


## Quick Start with DVC

### Initialize DVC (First Time)

```bash
make dvc-init
make install
```

### Run the Complete ML Pipeline

```bash
make dvc-repro
```

This runs all stages in order:
1. **Preprocess** — Raw data → Processed data
2. **Train** — Processed data → Model artifacts
3. **Evaluate** — Model evaluation and metrics

### View Pipeline Structure

```bash
make dvc-dag
```

For detailed setup, see [DVC_SETUP.md](DVC_SETUP.md).

## Model Containerization

✅ /predict endpoint exposed — Defined in serve_api.py as a POST endpoint at line 162.

✅ Accepts JSON input — Payload schema is HeartDiseaseInput with all 13 required features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal).

✅ Returns prediction + confidence — Response includes:

**prediction:** integer class (0 or 1 for heart disease)

**confidence:** float (max probability from model, 0–1 range)

**probabilities:** dict with per-class probabilities

**model_path:** path to the loaded model

✅ Container builds and runs locally — Your Dockerfile:

Uses Python 3.11-slim (efficient base)
Installs dependencies from requirements.txt
Runs as non-root user (appuser)
Exposes port 8000
Includes health check on /health
Integrated with DVC for artifact management

✅ Sample input ready — samples/predict_sample.json contains valid test data matching the schema exactly.

## Monitoring and Logging

The FastAPI serving app logs every API request as structured JSON and exposes Prometheus metrics.

Run the API with Prometheus and Grafana:

```bash
docker compose up --build
```

Open:

- API: http://localhost:8000
- Metrics endpoint: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana credentials:

- Username: `admin`
- Password: `admin`

The provisioned dashboard is named **Heart Disease API Monitoring**. It shows request rate, p95 latency, 5xx errors, and prediction count.

API request logs are written to:

```text
logs/api_requests.log
```

Useful metrics:

- `api_requests_total`
- `api_request_duration_seconds`
- `api_requests_in_progress`
- `api_predictions_total`

## Model containerization (recommended)

This repository can be containerized for local testing and production. Below is a recommended, production-friendly multi-stage Dockerfile and usage guidance.

Suggested Dockerfile (multi-stage, minimal image):

```dockerfile
# Stage 1: build
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install into a local prefix
COPY pyproject.toml poetry.lock requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM python:3.10-slim
WORKDIR /app

# Create an unprivileged user
RUN useradd --create-home appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code and model artifacts
COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# Start the app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
```

Build and tag the image locally:

```bash
docker build -t <your-username>/heart-disease-api:latest .
```

Run the container locally (mount model if needed):

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/model.pkl \
  <your-username>/heart-disease-api:latest
```

Push to a registry (Docker Hub example):

```bash
docker login
docker tag <your-username>/heart-disease-api:latest <your-username>/heart-disease-api:1.0.0
docker push <your-username>/heart-disease-api:1.0.0
```

Notes and best practices:

- Keep secrets out of the image. Use environment variables or a secret store (Vault, cloud secrets, or Kubernetes secrets).
- Pin dependency versions to ensure reproducible builds.
- Prefer a non-root user inside the container.
- Keep image layers small (clean apt caches, avoid unnecessary files).
- If your model file is large, consider mounting it as a volume or fetching it at startup from object storage (S3/GCS).
- Add health and readiness endpoints to the FastAPI app for orchestration checks (e.g., `/health`, `/ready`).
- Use DVC to version and manage large model artifacts.

## Deployment

Choose a deployment pattern based on scale and constraints. Below are recommended options and sample manifests/commands.

1) Local / Development: Docker Compose

Use docker-compose for local testing (already provided):

```bash
docker compose up --build
```

This starts the API, Prometheus, and Grafana as configured in `docker-compose.yml`.

2) CI/CD + Container Registry

Have your CI pipeline build and push images to a registry (Docker Hub, GHCR, ECR, GCR):

- Build image
- Run tests
- Run DVC pipeline
- Tag and push image
- Deploy (update k8s manifests or trigger rollout)

Example GitHub Actions outline:
- Checkout code
- Set up Python
- Install dependencies and run tests
- Run DVC pipeline (`dvc repro`)
- Build and push Docker image using `docker/build-push-action`
- Deploy to Kubernetes using `kubectl` or Helm

See `.github/workflows/dvc-pipeline.yaml` for a complete example.

3) Production: Kubernetes (recommended for production)

Add example manifests under `k8s/` (recommended files):

- k8s/deployment.yaml — Deployment pointing at image `<registry>/heart-disease-api:1.0.0`
- k8s/service.yaml — Service (ClusterIP / LoadBalancer) exposing port 8000
- k8s/hpa.yaml — HorizontalPodAutoscaler based on CPU or custom metrics
- k8s/ingress.yaml — Ingress for host routing and TLS via cert-manager

Apply manifests:

```bash
kubectl apply -f k8s/
```

Production recommendations:

- Configure liveness/readiness probes to `/health` and `/ready`.
- Set resource requests/limits for predictable scheduling.
- Store secrets in Kubernetes Secrets and configuration in ConfigMaps.
- Use an Ingress controller with TLS and, if needed, a service mesh for traffic management.
- Centralize logs (ELK, Grafana Loki) and metrics (Prometheus) with alerting.
- Pull DVC-tracked model artifacts during container initialization or deployment.

4) Serverless / PaaS options

- Google Cloud Run, AWS App Runner, or Azure App Service can run container images directly with minimal infra management. Push the image to a registry and deploy via the provider's CLI/console.

5) Safe rollouts

- Use rolling updates, health checks and readiness probes for zero-downtime deploys.
- Consider canary releases or blue/green deployments using service meshes or traffic-splitting tools.
- GitOps (ArgoCD / Flux) can be used to keep cluster state declarative.

## Data Versioning with DVC

DVC (Data Version Control) tracks large datasets and model artifacts:

- **Track data:** `dvc add data/raw data/processed`
- **Track models:** `dvc add models/`
- **Define pipelines:** Edit `dvc.yaml` with data/model dependencies
- **Run reproducible workflows:** `dvc repro`
- **Collaborate:** Push/pull artifacts from remote storage (S3, GCS, Azure, etc.)

For detailed guidance, see [DVC_SETUP.md](DVC_SETUP.md).

## Observability & Production-readiness Checklist

- Logging: structured JSON logs are written to `logs/api_requests.log`; consider shipping to a log aggregator
- Metrics: Prometheus metrics are exposed at `/metrics`
- Tracing: (optional) add OpenTelemetry to trace requests through the stack
- Health checks: implement `/live` and `/ready` endpoints
- Security: ensure images are scanned, run containers as non-root, and store secrets securely
- Backups: ensure model artifacts and important stateful data are versioned and backed up (using DVC)
- Data versioning: use DVC to track datasets and model lineage
- Reproducibility: use DVC pipelines to ensure experiments are reproducible

## Available Make Commands

```bash
make help              # Show all available commands
make install           # Install dependencies
make lint              # Run Ruff linter
make test              # Run pytest tests
make preprocess        # Run data preprocessing
make train             # Run model training
make serve             # Start FastAPI server
make docker-build      # Build Docker image
make docker-run        # Run container
make k8s-deploy        # Deploy to Kubernetes
make docker-compose-up # Start full stack
make dvc-init          # Initialize DVC
make dvc-repro         # Run DVC pipeline
make dvc-dag           # Show pipeline DAG
make dvc-push          # Push artifacts to remote
make dvc-pull          # Pull artifacts from remote
make clean             # Clean build artifacts
```

