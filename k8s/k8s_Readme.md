# Kubernetes Notes for This Repository

These manifests are tuned for the project's Dockerfile which runs the FastAPI app on port 8000 and exposes a `/health` endpoint.

## Before Applying

- Replace `heart-disease-api:1.0.0` in `k8s/deployment.yaml` with your image reference.
- Replace `example.com` in `k8s/ingress.yaml` with your host (or remove the Ingress).
- Create any needed secrets (docker registry, model credentials).

## Common Steps

1. **Create namespace:**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   ```

2. **Create docker registry secret (if needed):**
   ```bash
   kubectl create secret docker-registry regcred \
     --docker-server=<registry-server> \
     --docker-username=<username> \
     --docker-password=<password> \
     --docker-email=<email> \
     -n heart-disease-api
   ```

3. **(Optional) Create model PVC and copy model files, or mount a hostPath:**
   ```bash
   kubectl apply -f k8s/model-pvc.yaml
   ```
   Then update deployment to mount the PVC (see commented volumeMounts in deployment.yaml)

4. **Apply manifests:**
   ```bash
   kubectl apply -f k8s/
   ```

5. **To update the Deployment image:**
   ```bash
   kubectl -n heart-disease-api set image deployment/heart-disease-api \
     api=heart-disease-api:<tag>
   ```

## Notes About Probes and Model Loading

- **startupProbe** gives the container a long window (failureThreshold × periodSeconds) while the model loads; adjust these if your model loads faster/slower.
- If you implement a `/ready` endpoint that returns success only after the model is loaded, change the readinessProbe to use `/ready` — this prevents serving traffic before the model is available.

## Security & Permissions

- The Dockerfile creates a non-root user with UID 5678; the Deployment sets `runAsUser`/`runAsGroup` to 5678 and `fsGroup` to 5678 so mounted model files are readable.
- If your cluster enforces different Pod Security Standards, adjust the `securityContext` accordingly.
