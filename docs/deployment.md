# Deployment Guide

This guide covers deploying CodeAnalyzer in various environments, from development to production.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Deployment Options

### Quick Comparison

| Method | Best For | Complexity | Isolation | Scalability |
|--------|----------|------------|-----------|-------------|
| **Local** | Development, testing | Low | None | N/A |
| **Docker** | Consistency, portability | Medium | High | Medium |
| **Cloud VM** | Simple production | Medium | Medium | Low |
| **Kubernetes** | Enterprise, scale | High | High | High |

## Local Development

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool
- OpenAI API key

### Setup Steps

1. **Clone and setup:**
   ```bash
   git clone https://github.com/yourusername/codeanalyzer.git
   cd codeanalyzer

   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   pip install -e ".[dev]"
   ```

2. **Configure environment:**
   ```bash
   export OPENAI_API_KEY='your-api-key'
   export CODEANALYZER_LOG_LEVEL='INFO'
   ```

3. **Verify installation:**
   ```bash
   codeanalyzer --help
   pytest tests/
   ```

### Development Workflow

```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=codeanalyzer

# Lint and format
ruff check codeanalyzer/
black codeanalyzer/ tests/
mypy codeanalyzer/
```

## Docker Deployment

### Single Container

**Build and run:**
```bash
# Build image
docker build -t codeanalyzer:latest .

# Run interactively
docker run -it --rm \
  -e OPENAI_API_KEY='your-api-key' \
  -v $(pwd)/code-to-analyze:/code:ro \
  -v codeanalyzer-sessions:/app/sessions \
  -v codeanalyzer-logs:/app/logs \
  codeanalyzer:latest session new --name my-session

# Run one-off query
docker run --rm \
  -e OPENAI_API_KEY='your-api-key' \
  -v $(pwd):/code:ro \
  codeanalyzer:latest ask "Explain the main function"
```

### Docker Compose (Recommended)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  codeanalyzer:
    build: .
    image: codeanalyzer:latest
    container_name: codeanalyzer
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CODEANALYZER_LOG_LEVEL=${LOG_LEVEL:-INFO}
    volumes:
      # Your code to analyze (read-only)
      - ./your-project:/code:ro
      # Persistent data
      - codeanalyzer-sessions:/app/sessions
      - codeanalyzer-logs:/app/logs
      - codeanalyzer-config:/app/config
    stdin_open: true
    tty: true
    restart: unless-stopped

volumes:
  codeanalyzer-sessions:
  codeanalyzer-logs:
  codeanalyzer-config:
```

**Deploy:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Start service
docker-compose up -d

# Use the service
docker-compose exec codeanalyzer session new --name my-session
docker-compose exec codeanalyzer ask "How does authentication work?"

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Docker Production Configuration

**Dockerfile.prod:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (better caching)
COPY pyproject.toml README.md ./
COPY codeanalyzer/ ./codeanalyzer/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 codeanalyzer && \
    chown -R codeanalyzer:codeanalyzer /app

# Create directories
RUN mkdir -p /app/sessions /app/logs && \
    chown -R codeanalyzer:codeanalyzer /app/sessions /app/logs

USER codeanalyzer

ENV PYTHONUNBUFFERED=1
VOLUME ["/app/sessions", "/app/logs", "/code"]

WORKDIR /code
ENTRYPOINT ["codeanalyzer"]
CMD ["--help"]
```

**Build and push:**
```bash
# Build production image
docker build -f Dockerfile.prod -t codeanalyzer:prod .

# Tag for registry
docker tag codeanalyzer:prod registry.example.com/codeanalyzer:latest

# Push to registry
docker push registry.example.com/codeanalyzer:latest
```

## Cloud Deployment

### AWS EC2

**1. Launch EC2 instance:**
```bash
# Ubuntu 22.04 LTS, t3.medium or larger
# Open port 22 for SSH
# Attach IAM role with necessary permissions
```

**2. Install dependencies:**
```bash
# SSH into instance
ssh ubuntu@your-instance-ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**3. Deploy application:**
```bash
# Clone repository
git clone https://github.com/yourusername/codeanalyzer.git
cd codeanalyzer

# Set environment variables
echo "OPENAI_API_KEY=your-key" > .env

# Start with Docker Compose
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs
```

**4. Setup systemd service:**
```bash
# Create service file
sudo tee /etc/systemd/system/codeanalyzer.service > /dev/null <<EOF
[Unit]
Description=CodeAnalyzer Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/codeanalyzer
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable codeanalyzer
sudo systemctl start codeanalyzer
sudo systemctl status codeanalyzer
```

### Google Cloud Platform (GCP)

**1. Create Compute Engine instance:**
```bash
gcloud compute instances create codeanalyzer-vm \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-medium \
  --zone=us-central1-a \
  --tags=codeanalyzer
```

**2. Deploy with Container-Optimized OS:**
```bash
# Use Container-Optimized OS
gcloud compute instances create-with-container codeanalyzer-vm \
  --container-image=gcr.io/your-project/codeanalyzer:latest \
  --container-env=OPENAI_API_KEY=your-key \
  --machine-type=e2-medium \
  --zone=us-central1-a
```

### Azure

**1. Create Azure Container Instance:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name codeanalyzer \
  --image codeanalyzer:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables OPENAI_API_KEY=your-key \
  --azure-file-volume-account-name mystorageaccount \
  --azure-file-volume-share-name codeanalyzer-data \
  --azure-file-volume-mount-path /app/sessions
```

### Kubernetes

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codeanalyzer
  labels:
    app: codeanalyzer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: codeanalyzer
  template:
    metadata:
      labels:
        app: codeanalyzer
    spec:
      containers:
      - name: codeanalyzer
        image: codeanalyzer:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: codeanalyzer-secrets
              key: openai-api-key
        volumeMounts:
        - name: sessions
          mountPath: /app/sessions
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: sessions
        persistentVolumeClaim:
          claimName: codeanalyzer-sessions-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: codeanalyzer-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: codeanalyzer-service
spec:
  selector:
    app: codeanalyzer
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**Deploy:**
```bash
# Create secret
kubectl create secret generic codeanalyzer-secrets \
  --from-literal=openai-api-key=your-key

# Apply manifests
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl logs -f deployment/codeanalyzer
```

## Production Considerations

### Security

**1. API Key Management:**
```bash
# Use secret management services
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name codeanalyzer/openai-key \
  --secret-string your-api-key

# Retrieve in application
aws secretsmanager get-secret-value \
  --secret-id codeanalyzer/openai-key \
  --query SecretString \
  --output text
```

**2. Network Security:**
```bash
# Restrict access with firewall rules
# AWS Security Group
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 22 \
  --cidr your-ip/32
```

**3. Container Security:**
```dockerfile
# Run as non-root user
USER codeanalyzer

# Read-only filesystem
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /app/sessions \
  codeanalyzer:latest
```

### Performance Optimization

**1. Resource Allocation:**
```yaml
# Docker Compose resource limits
services:
  codeanalyzer:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

**2. Caching Strategy:**
```bash
# Use volume for pip cache
docker run \
  -v pip-cache:/root/.cache/pip \
  codeanalyzer:latest
```

**3. Session Management:**
```python
# Configure in config.json
{
  "max_sessions": 10,
  "session_ttl": 86400,  # 24 hours
  "cleanup_interval": 3600  # 1 hour
}
```

### Backup and Recovery

**1. Backup sessions:**
```bash
# Backup Docker volumes
docker run --rm \
  -v codeanalyzer-sessions:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar czf /backup/sessions-$(date +%Y%m%d).tar.gz /data
```

**2. Restore sessions:**
```bash
# Restore from backup
docker run --rm \
  -v codeanalyzer-sessions:/data \
  -v $(pwd)/backup:/backup \
  ubuntu tar xzf /backup/sessions-20231117.tar.gz -C /
```

**3. Automated backups:**
```bash
# Cron job for daily backups
0 2 * * * /usr/local/bin/backup-codeanalyzer.sh
```

## Monitoring and Maintenance

### Logging

**1. Centralized logging:**
```yaml
# Docker Compose with logging driver
services:
  codeanalyzer:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**2. Log aggregation:**
```bash
# Ship logs to ELK stack
docker run \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  codeanalyzer:latest
```

### Health Checks

**1. Docker health check:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD codeanalyzer --version || exit 1
```

**2. Kubernetes liveness probe:**
```yaml
livenessProbe:
  exec:
    command:
    - codeanalyzer
    - --version
  initialDelaySeconds: 30
  periodSeconds: 30
```

### Updates

**1. Rolling updates:**
```bash
# Pull new image
docker pull codeanalyzer:latest

# Restart with new image
docker-compose pull
docker-compose up -d
```

**2. Zero-downtime updates:**
```bash
# Kubernetes rolling update
kubectl set image deployment/codeanalyzer \
  codeanalyzer=codeanalyzer:v2.0
kubectl rollout status deployment/codeanalyzer
```

## Troubleshooting Deployment

### Common Issues

**1. Container won't start:**
```bash
# Check logs
docker logs codeanalyzer

# Check environment
docker exec codeanalyzer env

# Verify API key
docker exec codeanalyzer echo $OPENAI_API_KEY
```

**2. Permission issues:**
```bash
# Fix volume permissions
docker run --rm \
  -v codeanalyzer-sessions:/data \
  ubuntu chown -R 1000:1000 /data
```

**3. Out of memory:**
```bash
# Increase limits
docker update --memory 4g codeanalyzer
```

## CI/CD Integration

### GitHub Actions Deployment

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t codeanalyzer:latest .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push codeanalyzer:latest

      - name: Deploy to production
        run: |
          ssh deploy@server "cd /app && docker-compose pull && docker-compose up -d"
```

## Cost Optimization

**1. Use spot instances:**
- AWS EC2 Spot Instances
- GCP Preemptible VMs
- Azure Spot VMs

**2. Optimize image size:**
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
# Build dependencies

FROM python:3.11-slim
# Copy only runtime
```

**3. Resource right-sizing:**
```bash
# Monitor and adjust
docker stats codeanalyzer
# Adjust based on actual usage
```

## Support and Resources

- [Architecture Documentation](architecture.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Examples](examples.md)
- [GitHub Issues](https://github.com/yourusername/codeanalyzer/issues)
