#/bin/sh
set -e

. scripts/load_env.sh

# Start minikube only if not already running
if ! minikube status --format '{{.Host}}' 2>/dev/null | grep -q "Running"; then
  echo "🚀 Starting Minikube..."
  minikube start --cpus=6 --memory=16384 --driver=docker
else
  echo "✅ Minikube already running, skipping start."
fi

# Point Docker to use Minikube's Docker daemon
eval $(minikube docker-env)

# Build local images
docker build -t ai-service:latest ./ai-service
docker build -t agent-ui:latest ./agent-ui

# Helm repos
helm repo add milvus https://milvus-io.github.io/milvus-helm/ || true
helm repo update

kubectl create secret generic ai-service-secrets \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --from-literal=OPENAI_PROJECT_ID=$OPENAI_PROJECT_ID \
  --from-literal=GOOGLE_SEARCH_API_KEY=$GOOGLE_SEARCH_API_KEY \
  --from-literal=GOOGLE_SEARCH_CSE_ID=$GOOGLE_SEARCH_CSE_ID \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy Milvus
helm upgrade --install milvus milvus/milvus \
  -n milvus --create-namespace \
  -f k8s/milvus-local-values.yaml

# Deploy ai-service
helm upgrade --install ai-service ./helm/ai-service \
  --set registry="" \
  --set tag=latest

# Deploy agent-ui
helm upgrade --install agent-ui ./helm/agent-ui \
  --set registry="" \
  --set tag=latest
