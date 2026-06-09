# MLOps Assignment - Setup Guide

## Project Structure

```
MLops_assignment/
├── src/
│   ├── train.py              # Main training script
│   └── config.py             # Configuration management
├── data/
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── models/                   # Trained models
├── logs/                     # Training logs
├── outputs/                  # Training outputs
├── environment.yml           # Conda environment
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container
├── docker-compose.yml        # Docker compose orchestration
├── .env.example              # Environment variables template
└── README.md                 # Project documentation
```

## Setup Options

### Option 1: Local Setup with pip

**Prerequisites:**
- Python 3.9+
- pip

**Steps:**

```bash
# Clone the repository
git clone https://github.com/2024ac05306-droid/MLops_assignment.git
cd MLops_assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Run training
python src/train.py
```

### Option 2: Conda Environment Setup

**Prerequisites:**
- Conda/Miniconda installed

**Steps:**

```bash
# Clone the repository
git clone https://github.com/2024ac05306-droid/MLops_assignment.git
cd MLops_assignment

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate mlops-training

# Copy environment template
cp .env.example .env

# Run training
python src/train.py
```

### Option 3: Docker Setup (Recommended)

**Prerequisites:**
- Docker installed
- Docker Compose installed

**Steps:**

```bash
# Clone the repository
git clone https://github.com/2024ac05306-droid/MLops_assignment.git
cd MLops_assignment

# Build Docker image
docker build -t mlops-training:latest .

# Run container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           -v $(pwd)/outputs:/app/outputs \
           mlops-training:latest
```

### Option 4: Docker Compose (Full Stack)

**Prerequisites:**
- Docker installed
- Docker Compose installed

**Steps:**

```bash
# Clone the repository
git clone https://github.com/2024ac05306-droid/MLops_assignment.git
cd MLops_assignment

# Copy environment template
cp .env.example .env

# Build and start services
docker-compose up --build

# Access MLflow dashboard at http://localhost:5000
```

## Running the Training

### Method 1: Direct Python Execution
```bash
python src/train.py
```

### Method 2: With Docker
```bash
docker-compose run training python src/train.py
```

### Method 3: Docker Compose Full Stack
```bash
docker-compose up
```

## Configuration

Edit `.env` file to customize:
- Model parameters (batch size, epochs, learning rate)
- Data paths
- Output directories
- MLflow settings

Example:
```env
MODEL_TYPE=neural_network
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=0.0001
```

## Outputs

After training completes:
- **Model**: `models/model.pkl`
- **Scaler**: `models/scaler.pkl`
- **Metrics**: `outputs/metrics.txt`
- **Logs**: `logs/training.log`

## Monitoring

### TensorBoard
```bash
tensorboard --logdir=logs/tensorboard
```

### MLflow Dashboard
- Local: `http://localhost:5000`
- Docker: `http://localhost:5000`

## Troubleshooting

### Issue: Module not found
- **Solution**: Ensure all dependencies are installed
  ```bash
  pip install -r requirements.txt
  ```

### Issue: Permission denied for Docker
- **Solution**: Add user to docker group
  ```bash
  sudo usermod -aG docker $USER
  ```

### Issue: Port already in use (5000 or 6006)
- **Solution**: Change port in `docker-compose.yml` or stop conflicting services
  ```bash
  docker-compose down
  ```

## Next Steps

1. Add your own data to `data/raw/`
2. Modify `src/train.py` for your specific model
3. Update hyperparameters in `.env`
4. Set up CI/CD pipelines
5. Deploy to cloud platform

## Additional Resources

- [MLflow Documentation](https://mlflow.org/)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn Guide](https://scikit-learn.org/)
