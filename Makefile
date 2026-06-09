.PHONY: help install train docker-build docker-run docker-compose-up clean

help:
	@echo "MLOps Assignment - Available Commands"
	@echo "===================================="
	@echo "make install          - Install dependencies (pip)"
	@echo "make train            - Run model training locally"
	@echo "make docker-build     - Build Docker image"
	@echo "make docker-run       - Run training in Docker"
	@echo "make docker-compose-up - Start full stack with docker-compose"
	@echo "make clean            - Clean up generated files"
	@echo "make logs             - View training logs"

install:
	pip install -r requirements.txt

train:
	python src/train.py

docker-build:
	docker build -t mlops-training:latest .

docker-run:
	docker run -v $$(pwd)/data:/app/data \
	           -v $$(pwd)/models:/app/models \
	           -v $$(pwd)/logs:/app/logs \
	           -v $$(pwd)/outputs:/app/outputs \
	           mlops-training:latest

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
