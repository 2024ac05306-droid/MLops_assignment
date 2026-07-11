# Project Structure

```
MLops_assignment/
├── .github/
│   └── workflows/
├── data/
│   ├── .gitkeep
│   └── preprocessor.pkl
├── k8s/
│   ├── deployment.yaml
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   └── service.yaml
├── logs/
│   ├── .gitkeep
│   └── api_requests.log
├── models/
│   └── .gitkeep
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── api-monitoring-dashboard.json
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboard.yml
│           └── datasources/
│               └── prometheus.yml
├── outputs/
│   ├── eda/
│   ├── mlflow/
│   ├── predict/
│   └── reports/
├── samples/
│   └── predict_sample.json
├── src/
│   ├── config.py
│   ├── EDA_analysis.py
│   ├── model_inference.py
│   ├── model_train.py
│   ├── preprocess_data.py
│   └── serve_api.py
├── tests/
│   ├── conftest.py
│   ├── test_model_train.py
│   └── test_preprocess_data.py
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── compose.debug.yaml
├── compose.yaml
├── docker-compose.yml
├── Dockerfile
├── environment.yml
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
├── SERVING.md
├── SETUP_GUIDE.md
└── DEPLOYMENT.md
```
