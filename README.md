# MLops_assignment
This reporitory is  created for Mlops assignment. 

**Purpose:** An MLOps Experimental Learning Assignment focused on end-to-end ML model development with emphasis on:

1. ML model design and development
2. CI/CD pipelines and automation
3. Containerization and deployment
4. Experiment tracking
5. Cloud deployment and monitoring
6. Production-ready MLOps best practices


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
