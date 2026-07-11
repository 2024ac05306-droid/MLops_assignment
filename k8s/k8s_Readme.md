Kubernetes notes for this repository
These manifests are tuned for the project's Dockerfile which runs the FastAPI app on port 8000 and exposes a /health endpoint.

Before applying:

Replace /heart-disease-api:1.0.0 in k8s/deployment.yaml with your image reference.
Replace example.com in k8s/ingress.yaml with your host (or remove the Ingress).
Create any needed secrets (docker registry, model credentials).
Common steps:

Create namespace: kubectl apply -f k8s/namespace.yaml

Create docker registry secret (if needed): kubectl create secret docker-registry regcred
--docker-server= --docker-username= --docker-password= --docker-email= -n heart-disease-api

(Optional) Create model PVC and copy model files, or mount a hostPath: kubectl apply -f k8s/model-pvc.yaml

Then update deployment to mount the PVC (see commented volumeMounts in deployment.yaml)
Apply manifests: kubectl apply -f k8s/

To update the Deployment image: kubectl -n heart-disease-api set image deployment/heart-disease-api api=/heart-disease-api:tag

Notes about probes and model loading:

startupProbe gives the container a long window (failureThreshold × periodSeconds) while the model loads; adjust these if your model loads faster/slower.
If you implement a /ready endpoint that returns success only after the model is loaded, change the readinessProbe to use /ready — this prevents serving traffic before the model is available.
Security & permissions:

The Dockerfile creates a non-root user with UID 5678; the Deployment sets runAsUser/runAsGroup to 5678 and fsGroup to 5678 so mounted model files are readable. If your cluster enforces different PodSecurity policies, adapt the securityContext accordingly.