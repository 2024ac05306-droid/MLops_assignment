# Model Serving API

This project serves the trained heart disease model with FastAPI.

## Build the Docker image

```bash
docker build -t mlops-model-api:latest .
```

## Run the container

```bash
docker run --rm -p 8000:8000 mlops-model-api:latest
```

The API will be available at:

```text
http://localhost:8000
```

## Health check

```bash
curl http://localhost:8000/health
```

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
