#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Build and Deploy to Kubernetes (Local with Minikube)
# ------------------------------------------------------------------------------

# Generate a unique image tag (use git commit SHA if available, otherwise timestamp)
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  IMAGE_TAG=$(git rev-parse HEAD)
else
  IMAGE_TAG=$(date +%s)
fi
echo "Using image tag: $IMAGE_TAG"

# ------------------------------------------------------------------------------
# Build images inside Minikube’s Docker daemon
# ------------------------------------------------------------------------------
echo ">> Building Docker images inside Minikube…"
eval $(minikube -p mychatbot docker-env)

docker build -t ai-service:${IMAGE_TAG} ./ai-service
docker build -t agent-ui:${IMAGE_TAG} ./agent-ui

# ------------------------------------------------------------------------------
# Deploy Milvus with Helm
# ------------------------------------------------------------------------------
echo ">> Deploying Milvus (local Helm)…"
helm repo add milvus https://milvus-io.github.io/milvus-helm/ || true
helm repo update
helm upgrade --install milvus milvus/milvus \
  --namespace milvus --create-namespace \
  -f k8s/milvus-local-values.yaml

# ------------------------------------------------------------------------------
# Deploy ai-service
# ------------------------------------------------------------------------------
echo ">> Deploying ai-service…"
helm upgrade --install ai-service ./helm/ai-service \
  --set registry=ai-service \
  --set tag=${IMAGE_TAG}

# ------------------------------------------------------------------------------
# Deploy agent-ui
# ------------------------------------------------------------------------------
echo ">> Deploying agent-ui…"
helm upgrade --install agent-ui ./helm/agent-ui \
  --set registry=agent-ui \
  --set tag=${IMAGE_TAG}

echo "✅ Local build and deploy completed successfully!"
