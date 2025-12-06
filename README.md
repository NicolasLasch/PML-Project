# PML-Project

# ML Bike Prediction API 🚴‍♂️

[![App Runner](https://img.shields.io/badge/App%20Runner-Live-green)](https://yfpzcgnbsf.ap-southeast-1.awsapprunner.com/)
[![ECR](https://img.shields.io/badge/ECR-Private-blue)](https://console.aws.amazon.com/ecr/private)
[![Docker](https://img.shields.io/badge/Docker-Published-blue)](https://hub.docker.com/r/doubleyou02/bergen-bike-api)

## Architecture

Local → Docker Build → Private ECR → App Runner → HTTPS Global API


## Deployment Steps (AWS Private ECR)

### 1. Docker Build & Test

Build ML model container

`docker build -t bergen-bike-api:latest .`

Test locally (Flask on port 8000)

`docker run -p 8000:8000 bergen-bike-api:latest`

### 2. Private ECR Repository

AWS Console → ECR → "Create repository" → Set Repo name → "Create"

### 3. Push Image to Private ECR

Commands can be found under ECR in the:

AWS Console → Private Repository created → Images → View Push Commands (macOS, Linux, Windows based)

(Commands below are Linux-based)

#### 1. Docker login to ECR
`aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com`

#### 2. Tag local image for ECR
`docker tag bergen-bike-api:latest <AWS_ACCOUTN_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-pred-app:latest`

#### 3. Push to ECR
`docker push <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-pred-app:latest`

### 4. Deploy to App Runner

AWS Console → App Runner → "Create an App Runner service"

#### Service Configuration
Service name: `bergen-bike-api` (any name preferred)

#### Deployment Source
☑️ Container Registry

Image URI: `<AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/bergen-bike-pred-app:latest`
(private container image URI pushed to ECR previously)

Container port: `8000`

→ "Next" → "Create & deploy"

### 5. Access Live API

