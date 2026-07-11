# Model Serving API

This project serves the trained heart disease model with FastAPI.

## Build the Docker image

```bash
docker build -t mlops-model-api:latest .
```


## Run the container (recommended: mount models directory)

If you have a trained model at ./models/best_pipeline.pkl, mount the models
folder into the container so the server can find it at runtime:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" -v "$(pwd)/logs:/app/logs" \
  -e MODEL_PATH=models/best_pipeline.pkl mlops-model-api:latest
```

If you prefer to bake the model into the image, copy it into the `models/`
folder before building the image.

## Run the full stack with compose (serving + prometheus + grafana)

```bash
docker compose -f compose.yaml up --build
```

Service URLs when running the compose stack:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)


## Health check

The container exposes a health endpoint `/health`. The endpoint will return
HTTP 200 only after the model has been successfully loaded. This enables
orchestrators (Docker, Kubernetes) to detect readiness correctly.

```bash
curl http://localhost:8000/health
```

If the model is missing the endpoint returns HTTP 503 with details.

## Sample prediction

Linux/macOS/Git Bash:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @samples/predict_sample.json
```

PowerShell:

```powershell
$body = Get-Content samples/predict_sample.json -Raw
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Expected response shape:

```json
{
  "prediction": 0,
  "confidence": 0.873421,
  "probabilities": {
    "0": 0.873421,
    "1": 0.126579
  },
  "model_path": "models/best_pipeline.pkl"
}
```

Prediction values:

- `0`: no heart disease predicted
- `1`: heart disease predicted
