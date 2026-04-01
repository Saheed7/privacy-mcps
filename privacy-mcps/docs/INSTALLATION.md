# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 / Windows 10 / macOS 12 | Ubuntu 22.04 LTS |
| Python | 3.10 | 3.10+ |
| RAM | 16 GB | 64 GB |
| GPU | NVIDIA GTX 1080 (8 GB) | NVIDIA RTX 3090 (24 GB) |
| CUDA | 11.8 | 12.0+ |
| Storage | 20 GB free | 50 GB free |
| Docker | 24.0+ (optional) | 24.0+ |
| Kubernetes | 1.27+ (optional) | 1.27+ |

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/privacy-mcps.git
cd privacy-mcps
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify GPU (Optional)
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 5. Download Datasets

**CIC-IoMT2024:**
1. Visit https://www.unb.ca/cic/datasets/
2. Download the CIC-IoMT2024 dataset
3. Place the CSV file in `data/CIC-IoMT2024.csv`

**Edge-IIoTset:**
1. Visit https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
2. Download and extract
3. Place the CSV file in `data/Edge-IIoTset.csv`

### 6. Run Tests
```bash
python -m pytest tests/ -v
```

### 7. Docker Setup (Optional)
```bash
cd docker
docker-compose up --build
```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| CUDA out of memory | Reduce `batch_size` in `configs/config.yaml` |
| phe import error | `pip install phe` |
| gRPC proto not compiled | See `src/microservices/mlams.py` for instructions |
| Dataset not found | Ensure CSV files are in `data/` directory |
