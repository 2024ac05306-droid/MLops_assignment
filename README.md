# MLops_assignment
This reporitory is  created for Mlops assignment. 

**Purpose:** An MLOps Experimental Learning Assignment focused on end-to-end ML model development with emphasis on:

1. ML model design and development
2. CI/CD pipelines and automation
3. Containerization and deployment
4. Experiment tracking
5. Cloud deployment and monitoring
6. Production-ready MLOps best practices

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
