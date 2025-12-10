# PML-Project

# 🚴‍♂️ Bergen Bike Prediction API 

[![App Runner](https://img.shields.io/badge/📱App%20Runner-Live-green)](https://yfpzcgnbsf.ap-southeast-1.awsapprunner.com/) [![Docker](https://img.shields.io/badge/🐋Docker-Published-blue)](https://hub.docker.com/r/doubleyou02/bergen-bike-api) [![CI/CD](https://img.shields.io/badge/♾️CI%2FCD-GitHub%20Actions-brightgreen)](https://github.com/NicolasLasch/PML-Project/actions)

## 🔎 Overview
Bergen Bike Prediction API is a Flask-based machine learning service that predicts bike rental demand at Bergen stations given features like time, weather, and station metadata. The project is containerized with Docker and deployed via GitHub Actions to a private AWS ECR repository and served globally using AWS App Runner.

## 📊 Dataset

Bergen Bike Sharing Dataset Source : 

`https://www.kaggle.com/datasets/amykzhang/bergen-bike-sharing-dataset-2023`

## 🧩 Architecture

Local → Docker Build → Private ECR → App Runner → HTTPS Global API

**Project Structure**:
```
|_ app.py – Flask API entrypoint
|_ src/ – data processing, feature engineering, model training/evaluation, station recommendation utilities
|_ models/ – serialized trained model artifacts
|_ data/ – raw and processed example data
|_ Dockerfile – container build recipe
|_ tests_all.py - consolidated unit + integration test suite
```

## 🛠️ Local Development
**Requirements**
- Python
- Docker (optional but recommended)
- `pip` for Python dependencies

**Setup with Virtual environment (recommended)**

From project root:

(macOS/Linux)
```bash
python3 -m venv venv
source venv/bin/activate
```

(Windows)
```bash
python3 -m venv venv
venv\Scripts\activate
```
**Download dependencies**

```bash
pip install -r requirements.txt
```

**Run without Docker**

```bash
python app.py  # serves on http://localhost:8000
```

**Build and run with Docker**

```bash
# Build image
docker build -t bergen-bike-api:latest .

# Run container locally (maps host 8000 → container 8000)
docker run -p 8000:8000 bergen-bike-api:latest
```

## 📦 Deployment (AWS Private ECR, AppRunner)

**1. Docker Build & Test**

Build ML model container

```bash
docker build -t bergen-bike-api:latest .
```

Test locally (Flask on port 8000)

```bash
docker run -p 8000:8000 bergen-bike-api:latest
```

**2. Private ECR Repository**

```
AWS Console → ECR → "Create repository" → Set Repo name → "Create"
```

**3. Push Image to Private ECR**

Commands can be found under ECR in the:

```
AWS Console → Private Repository created → Images → View Push Commands (macOS, Linux, Windows based)
```

#### 1. Docker login to ECR

- (macOS/Linux)
```bash
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com`
```
- (Windows)
```bash
(Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com
```

#### 2. Tag local image for ECR

```bash
docker tag bergen-bike-api:latest <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-api:latest
```

#### 3. Push to ECR

```bash
docker push <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-api:latest
```

**4. Deploy to App Runner**
```
AWS Console → App Runner → "Create an App Runner service"
```

#### Service Configuration

Service name: `bergen-bike-api` (any name preferred)

#### Deployment Source

☑️ Container Registry

Image URI: `<AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-api:latest` (private container image URI pushed to ECR previously)

Container port: `8000`
```
→ "Next" → "Create & deploy"
```
**5. Access Live API**

Your API is now live at: `https://yfpzcgnbsf.ap-southeast-1.awsapprunner.com/`

---

## 🧪 Testing

Run the test suite:

```bash
pytest test_all.py -v
```

**Test Coverage:**

* Data processing pipeline
* Feature engineering functions
* Model training & evaluation
* Station recommendation system
* API endpoints (Flask)

## 🚀 CI/CD Pipeline

**Automated deployment on every push to `main`:**

```
Push to GitHub → Run Tests → Build Docker → Push to ECR → Deploy to App Runner
```

**What happens automatically:**

* ✅ **Continuous Integration**: Runs 22+ unit tests + code linting
* ✅ **Continuous Deployment**: Builds Docker image → Pushes to ECR → Updates App Runner
* ⏱️ **Total time**: \~3 minutes from push to live

**How to use:**

```bash
git add .
git commit -m "feat: add new feature"
git push origin main
# ☕ That's it! GitHub Actions handles the rest
```
