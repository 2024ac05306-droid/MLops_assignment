# Production Deployment

This project can be deployed to local Kubernetes using Docker Desktop Kubernetes or Minikube.

The deployment uses:

- Docker image: `mlops-model-api:latest`
- Kubernetes Deployment: `heart-disease-api`
- Kubernetes Service: `heart-disease-api-service`
- Service type: `LoadBalancer`
- API endpoint: `/predict`
- Health endpoint: `/health`

## 1. Build the Docker image

```bash
docker build -t mlops-model-api:latest .
```

## 2. Deploy to Kubernetes

```bash
kubectl apply -k k8s
```

## 3. Verify Kubernetes resources

```bash
kubectl get all -n mlops-assignment
kubectl get pods -n mlops-assignment
kubectl get service heart-disease-api-service -n mlops-assignment
```

Wait until the pod status is `Running` and ready.

## 4. Access the API

### Docker Desktop Kubernetes

Docker Desktop normally exposes the LoadBalancer on `localhost`.

```bash
curl http://localhost/health
```

Prediction:

```bash
curl -X POST "http://localhost/predict" \
  -H "Content-Type: application/json" \
  --data @samples/predict_sample.json
```

### Minikube

If using Minikube, either run a tunnel:

```bash
minikube tunnel
```

Then use the external IP shown by:

```bash
kubectl get service heart-disease-api-service -n mlops-assignment
```

Or open a local service URL:

```bash
minikube service heart-disease-api-service -n mlops-assignment
```

## 5. PowerShell endpoint verification

Health:

```powershell
Invoke-RestMethod -Uri "http://localhost/health"
```

Prediction:

```powershell
$body = Get-Content samples/predict_sample.json -Raw
Invoke-RestMethod -Uri "http://localhost/predict" -Method Post -ContentType "application/json" -Body $body
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

## 6. Screenshots to include in submission

Capture these screenshots for the deployment marks:

1. Docker image built successfully:

```bash
docker images mlops-model-api
```

2. Kubernetes deployment and pod running:

```bash
kubectl get all -n mlops-assignment
```

3. LoadBalancer service exposed:

```bash
kubectl get service heart-disease-api-service -n mlops-assignment
```

4. Health endpoint success:

```bash
curl http://localhost/health
```

5. Prediction endpoint success:

```bash
curl -X POST "http://localhost/predict" -H "Content-Type: application/json" --data @samples/predict_sample.json
```

## 7. Clean up

```bash
kubectl delete -k k8s
```
