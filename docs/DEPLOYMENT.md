# Deployment Guide

Production deployment, scaling, and operational considerations for DocBases.

## Deployment Environments

### Development

**Setup**: Local machine with Ollama
**LLM**: Free local models (llama3.1:8b)
**Storage**: File-based ChromaDB
**Observability**: File logs only

```env
LLM_PROVIDER=ollama
EMB_PROVIDER=ollama
USE_PERSISTENT_MEMORY=true
LANGSMITH_TRACING=false
LOG_LEVEL=INFO
```

### Staging

**Setup**: Single server or container
**LLM**: Self-hosted or API (low cost)
**Storage**: Persistent database
**Observability**: Basic metrics

```env
LLM_PROVIDER=groq              # Fast, free tier available
EMB_PROVIDER=openai            # Good quality
USE_PERSISTENT_MEMORY=true
LANGSMITH_TRACING=true
LOG_LEVEL=INFO
```

### Production

**Setup**: Containerized, load-balanced
**LLM**: Enterprise API (OpenAI, Google)
**Storage**: Managed database service
**Observability**: Full tracing and monitoring

```env
LLM_PROVIDER=openai
EMB_PROVIDER=openai
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=/var/data/checkpoints.db
METRICS_DB_PATH=/var/data/metrics.db
LANGSMITH_TRACING=true
LOG_LEVEL=WARNING
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p knowledges logs temps

# Expose port (if using API mode)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run application
CMD ["python", "src/main.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  doc-bases:
    build: .
    environment:
      LLM_PROVIDER: openai
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      EMB_PROVIDER: openai
      RAG_MODE: adaptive
      USE_PERSISTENT_MEMORY: "true"
      LANGSMITH_TRACING: "true"
      LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
    volumes:
      - ./knowledges:/app/knowledges
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Local Ollama for embeddings fallback
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./ollama-data:/root/.ollama
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t doc-bases:latest .

# Run container
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e LANGSMITH_API_KEY=$LANGSMITH_API_KEY \
           -v $(pwd)/knowledges:/app/knowledges \
           -v $(pwd)/logs:/app/logs \
           doc-bases:latest

# With compose
docker-compose up -d
```

## Kubernetes Deployment

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: docbases-config
  namespace: docbases
data:
  LLM_PROVIDER: "openai"
  EMB_PROVIDER: "openai"
  RAG_MODE: "adaptive"
  USE_PERSISTENT_MEMORY: "true"
  LOG_LEVEL: "INFO"
  KNOWLEDGE_BASE_DIR: "/data/knowledges"
  CHECKPOINT_DB_PATH: "/data/checkpoints.db"
  METRICS_DB_PATH: "/data/metrics.db"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: docbases-secrets
  namespace: docbases
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  LANGSMITH_API_KEY: "ls_..."
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docbases
  namespace: docbases
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docbases
  template:
    metadata:
      labels:
        app: docbases
    spec:
      containers:
      - name: docbases
        image: doc-bases:latest
        imagePullPolicy: IfNotPresent
        envFrom:
        - configMapRef:
            name: docbases-config
        - secretRef:
            name: docbases-secrets
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: docbases-pvc
```

### PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: docbases-pvc
  namespace: docbases
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
```

## Database Setup

### PostgreSQL (For Checkpointing at Scale)

```sql
-- Alternative to SQLite for production
CREATE DATABASE docbases;

CREATE TABLE checkpoints (
  thread_id TEXT NOT NULL,
  checkpoint_id TEXT NOT NULL,
  parent_checkpoint_id TEXT,
  type_id TEXT NOT NULL,
  checkpoint BYTEA NOT NULL,
  metadata BYTEA,
  created_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX idx_thread_id ON checkpoints(thread_id);
CREATE INDEX idx_created_at ON checkpoints(created_at);
```

### MongoDB (For Flexible Metrics)

```javascript
// Metrics collection
db.createCollection("query_metrics");

db.query_metrics.createIndex({ timestamp: 1 });
db.query_metrics.createIndex({ rag_mode: 1 });
db.query_metrics.createIndex({ session_id: 1 });

// Sample document
{
  _id: ObjectId(...),
  timestamp: ISODate("2026-01-16T10:30:00Z"),
  query: "What is RAG?",
  latency_ms: 1240,
  retrieval_count: 5,
  rag_mode: "adaptive",
  session_id: "sess_123",
  success: true
}
```

## Configuration Management

### Environment-Based Configuration

**Development**:
```bash
cp .env.example .env
# Edit with local settings
```

**Staging**:
```bash
# Use environment variables
export LLM_PROVIDER=groq
export LANGSMITH_TRACING=true
python src/main.py
```

**Production**:
```bash
# Docker/K8s secrets
kubectl create secret generic docbases-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=LANGSMITH_API_KEY=ls_...
```

### Secrets Management

**Using HashiCorp Vault**:
```python
import hvac

client = hvac.Client(url='https://vault.example.com', token='s.xxx')
secret = client.secrets.kv.read_secret_version(path='docbases')
api_key = secret['data']['data']['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = api_key
```

**Using AWS Secrets Manager**:
```python
import boto3
import json

client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='docbases-secrets')
credentials = json.loads(secret['SecretString'])
os.environ['OPENAI_API_KEY'] = credentials['OPENAI_API_KEY']
```

## Monitoring & Observability

### Logging Configuration

```python
# src/utils/logger.py integration
import logging
import json
from pythonjsonlogger import jsonlogger

# JSON logging for production
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
```

**Log Aggregation**: Send to ELK, Datadog, or CloudWatch

### Metrics Collection

```python
from src.observability.metrics import get_metrics_tracker

metrics = get_metrics_tracker()
metrics.log_query(
    query=query,
    latency_ms=duration_ms,
    rag_mode=mode,
    session_id=session_id,
    success=success
)
```

**Metrics Dashboard**: Query `knowledges/metrics.db` or export to Prometheus

### LangSmith Tracing

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls_xxx
LANGSMITH_PROJECT=doc-bases-prod
```

View traces at [smith.langchain.com](https://smith.langchain.com)

### Health Checks

Implement health endpoints:

```python
@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness check."""
    # Verify DB connections, LLM availability
    return {"ready": True}
```

## Scaling Considerations

### Horizontal Scaling

**Load Balancing**:
- Use nginx or cloud load balancer
- Route requests to multiple instances
- Sticky sessions for conversation continuity

**Shared Storage**:
- Use managed database service (PostgreSQL, MongoDB)
- Shared NFS or object storage (S3, GCS)
- Redis for distributed caching

### Vertical Scaling

**Hardware Requirements**:
- 2GB+ RAM per instance
- 2+ CPU cores
- SSD for database operations

**Optimization**:
- Use faster LLM providers (Groq, OpenAI)
- Increase embedding cache
- Optimize chunk sizes

### Cost Optimization

```env
# Use free tier providers where possible
LLM_PROVIDER=groq              # Free tier: 500 requests/day
EMB_PROVIDER=ollama            # Self-hosted, free

# Or mix providers
# Simple queries: Groq (cheap)
# Complex queries: OpenAI (better quality)
```

## Backup & Recovery

### Database Backups

```bash
# SQLite backup
cp knowledges/checkpoints.db knowledges/checkpoints.db.backup

# Or use automated backup service
# AWS RDS, Google Cloud SQL, etc.
```

### Knowledge Base Backups

```bash
# Backup ChromaDB
tar -czf knowledges_backup.tar.gz knowledges/

# Restore
tar -xzf knowledges_backup.tar.gz
```

### Disaster Recovery

```bash
# Keep backups in separate location
aws s3 cp knowledges_backup.tar.gz s3://backup-bucket/

# Regular restore testing
aws s3 cp s3://backup-bucket/knowledges_backup.tar.gz .
tar -xzf knowledges_backup.tar.gz
```

## Performance Tuning

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_model():
    """Cache embedding model."""
    return get_embedding_model()
```

### Query Optimization

```env
# Tune retrieval parameters
RETRIEVAL_K=5              # Number of documents
RETRIEVAL_FETCH_K=20       # Pre-filter pool
RETRIEVAL_LAMBDA_MULT=0.25 # Diversity parameter

# Tune chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Resource Limits

```yaml
# Kubernetes resource requests/limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"
```

## Troubleshooting Deployments

### Common Issues

**Container won't start**:
```bash
# Check logs
docker logs container_id

# Verify environment variables
docker inspect container_id
```

**Memory leaks**:
```bash
# Monitor memory usage
docker stats container_id

# Profile memory
python -m memory_profiler src/main.py
```

**Database lock**:
```bash
# SQLite lock issues
rm -f knowledges/.chroma_lock

# Switch to PostgreSQL for concurrency
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_TYPE=postgresql
```

**LLM timeout**:
```python
# Increase timeouts
LLM_REQUEST_TIMEOUT=60  # seconds
```

## Post-Deployment

### Monitoring Checklist

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] LangSmith traces appearing
- [ ] Database backups running
- [ ] Log aggregation working
- [ ] Alerts configured

### Optimization Cycle

1. **Monitor**: Collect metrics for 1-2 weeks
2. **Analyze**: Identify bottlenecks
3. **Optimize**: Adjust configuration
4. **Measure**: Verify improvements

### Regular Maintenance

```bash
# Weekly
- Review metrics dashboard
- Check disk space
- Verify backups

# Monthly
- Clean old sessions (> 30 days)
- Archive logs
- Update dependencies
- Security patches

# Quarterly
- Load testing
- Disaster recovery drill
- Cost analysis
```

---

**Last Updated**: January 2026
**Deployment Version**: 2.0
