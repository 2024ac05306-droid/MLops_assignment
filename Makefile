.PHONY: help install lint test preprocess train serve docker-build docker-run k8s-deploy k8s-status k8s-delete docker-compose-up clean

help:
	@echo "MLOps Assignment - Available Commands"
	@echo "===================================="
	@echo "make install          - Install dependencies (pip)"
	@echo "make lint             - Run Ruff lint checks"
	@echo "make test             - Run unit tests"
	@echo "make preprocess       - Run data preprocessing"
	@echo "make train            - Run model training locally"
	@echo "make serve            - Run FastAPI model serving API locally"
	@echo "make docker-build     - Build Docker image"
	@echo "make docker-run       - Run training in Docker"
	@echo "make k8s-deploy       - Deploy API to local Kubernetes"
	@echo "make k8s-status       - Show Kubernetes deployment status"
	@echo "make k8s-delete       - Delete Kubernetes deployment"
	@echo "make docker-compose-up - Start full stack with docker-compose"
	@echo "make clean            - Clean up generated files"
	@echo "make logs             - View training logs"

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest

preprocess:
	python src/preprocess_data.py

train:
	python src/model_train.py

serve:
	uvicorn src.serve_api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t mlops-model-api:latest .

docker-run:
	docker run --rm -p 8000:8000 \
	           -v $$(pwd)/models:/app/models \
	           mlops-model-api:latest

k8s-deploy:
	kubectl apply -k k8s

k8s-status:
	kubectl get all -n mlops-assignment

k8s-delete:
	kubectl delete -k k8s

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

logs:
	tail -f logs/training.log

clean:
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
