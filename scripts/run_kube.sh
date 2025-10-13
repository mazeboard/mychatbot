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
docker build -t mychatbot:latest ./

# Helm repos
helm repo add milvus https://milvus-io.github.io/milvus-helm/ || true
helm repo update

kubectl create secret generic mychatbot-secrets \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --from-literal=OPENAI_PROJECT_ID=$OPENAI_PROJECT_ID \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy Milvus
helm upgrade --install milvus milvus/milvus \
  -n milvus --create-namespace \
  -f k8s/milvus-local-values.yaml

# Deploy mychatbot
helm upgrade --install mychatbot ./helm \
  --set registry="" \
  --set tag=latest
