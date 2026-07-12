# DVC Setup Guide

This guide explains how to configure and use DVC (Data Version Control) in this MLOps project.

## What is DVC?

DVC is a version control system for machine learning projects. It helps you:
- Track data and model artifacts without storing them in Git
- Define reproducible ML pipelines
- Manage experiments and their dependencies
- Collaborate on ML workflows with remote storage

## Quick Start

### 1. Initialize DVC (First Time Only)

```bash
make dvc-init
# or manually
dvc init --no-scm
```

This creates:
- `.dvc/` directory with configuration
- `.dvc/config` file for settings
- `.dvcignore` file (similar to `.gitignore`)

### 2. Install DVC Dependencies

```bash
make install
# Installs dvc==3.48.0 and dvc-s3==3.2.0
```

### 3. Configure Remote Storage

DVC requires remote storage for large artifacts (datasets, models). Choose one:

#### Option A: Local Storage (Development)

```bash
dvc remote add -d local /mnt/dvc-storage
# or
dvc remote modify local url /mnt/dvc-storage
```

#### Option B: AWS S3

```bash
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote region us-east-1
# Configure AWS credentials in ~/.aws/credentials or environment
```

#### Option C: Google Cloud Storage

```bash
dvc remote add -d myremote gs://my-bucket/dvc-storage
# Configure GCS credentials
```

#### Option D: Azure Blob Storage

```bash
dvc remote add -d myremote azure://my-container/dvc-storage
# Configure Azure credentials
```

### 4. View Current Configuration

```bash
dvc remote list
cat .dvc/config
```

## Pipeline Stages

The `dvc.yaml` file defines three stages:

### Stage 1: Preprocess
- **Input:** Raw data from `data/raw/`
- **Output:** Processed data in `data/processed/`
- **Script:** `src/preprocess_data.py`
- **Logs:** `logs/preprocess.log`

```bash
dvc repro preprocess
```

### Stage 2: Train
- **Input:** Processed data from `data/processed/`
- **Outputs:** 
  - Model: `models/model.pkl`
  - Metadata: `outputs/model_metadata.json`
- **Metrics:** `outputs/metrics.json`
- **Script:** `src/model_train.py`
- **Logs:** `logs/training.log`

```bash
dvc repro train
```

### Stage 3: Evaluate
- **Input:** Trained model and processed data
- **Outputs:** Evaluation metrics and plots
- **Script:** `src/evaluate_model.py`
- **Metrics:** `outputs/evaluation_metrics.json`
- **Logs:** `logs/evaluation.log`

```bash
dvc repro evaluate
```

## Common DVC Commands

### Run the Full Pipeline

```bash
make dvc-repro
# or
dvc repro
```

Runs all stages in dependency order. If no data changed, DVC skips already-completed stages.

### View Pipeline DAG

```bash
make dvc-dag
# or
dvc dag
```

Shows the pipeline dependency graph as text.

### Track Data/Models

```bash
# Track large directories
dvc add data/raw
dvc add models

# Generates data/raw.dvc and models.dvc files
# Commit .dvc files to Git, not the data itself
git add data/raw.dvc models.dvc
git commit -m "Add DVC tracking for data and models"
```

### Push/Pull Artifacts

```bash
# Push to remote storage
make dvc-push
# or
dvc push

# Pull from remote storage
make dvc-pull
# or
dvc pull
```

### Check Pipeline Status

```bash
dvc status
```

Shows which stages are out of date (dependencies changed but stage not re-run).

### View Metrics

```bash
dvc metrics show
dvc plots show
```

### Run a Specific Stage

```bash
dvc repro preprocess     # Runs only preprocessing
dvc repro train          # Runs training and its dependencies
```

## Integration with Docker & Kubernetes

### In Dockerfile

No changes needed! DVC files (`.dvc`) and configs are tracked in Git. Large artifacts are fetched from remote storage during build if needed:

```dockerfile
# Add to Dockerfile if models are tracked by DVC
RUN dvc pull models/
```

### In Kubernetes

Add init container or volume to pull artifacts:

```yaml
initContainers:
  - name: dvc-pull
    image: python:3.11-slim
    command:
      - sh
      - -c
      - |
        pip install dvc dvc-s3
        dvc pull models/
    volumeMounts:
      - name: models
        mountPath: /app/models
```

## Integration with CI/CD (GitHub Actions)

Example workflow:

```yaml
name: ML Pipeline
on: [push, pull_request]

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run DVC pipeline
        run: dvc repro
      
      - name: Push metrics
        run: dvc push
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

## Integration with MLflow

DVC and MLflow complement each other:

| Tool | Purpose |
|------|---------|
| **DVC** | Tracks data, models, and pipeline dependencies. Ensures reproducibility. |
| **MLflow** | Tracks experiments, hyperparameters, metrics. Enables comparison and registration. |

**Workflow:**
1. DVC ensures data/code reproducibility
2. MLflow logs experiments and metrics
3. MLflow model registry stores production models
4. DVC tracks which data/code version created each model

## Troubleshooting

### DVC Not Initialized

```bash
# Check if .dvc directory exists
ls -la .dvc/

# Re-initialize if missing
make dvc-init
```

### Pipeline Failed

```bash
# Check status
dvc status

# View error logs
cat logs/training.log

# Re-run with debug
dvc repro --verbose
```

### Remote Storage Not Configured

```bash
# List configured remotes
dvc remote list

# Add a remote
dvc remote add -d myremote <path-or-url>

# Verify
dvc remote list
```

### Cannot Push/Pull

```bash
# Check remote credentials/access
dvc remote validate

# Try with verbose output
dvc push --verbose
dvc pull --verbose
```

### Clear DVC Cache

```bash
# Remove all cached files
rm -rf .dvc/cache

# or prune old unused cache
dvc gc
```

## Best Practices

1. **Commit `.dvc` files to Git**, not large data files
2. **Use meaningful stage names** in `dvc.yaml` for clarity
3. **Set up remote storage early** to enable team collaboration
4. **Run `dvc status` before commits** to ensure pipeline is clean
5. **Use `.dvcignore`** to exclude files (similar to `.gitignore`)
6. **Pin DVC version** in `requirements.txt` for reproducibility
7. **Document pipeline stages** in `dvc.yaml` with clear dependencies
8. **Use metrics/plots** for experiment comparison

## References

- [DVC Official Documentation](https://dvc.org/doc)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [DVC Remote Storage](https://dvc.org/doc/user-guide/data-management/remote)
- [DVC with MLflow](https://dvc.org/blog/ml-experiment-tracking-dvc-mlflow)
