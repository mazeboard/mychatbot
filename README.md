# mychatbot

Run Milvus, ai-service, and agent-ui all in the same cluster.

Each component is isolated by namespace (milvus, chatbot, etc.), not by cluster.

Advantages:

Easier to manage (kubectl apply -f k8s/ once).

In-cluster networking is automatic (e.g. ai-service just connects to milvus:19530).

Cheaper — fewer resources needed.

This is how most teams start.

## IP of localhost

ip route|grep default
default via 192.168.1.1 dev wlo1 proto dhcp src `192.168.1.140` metric 600

If you want login/push/pull to work cleanly, add your host registry as an insecure registry:

Edit or create /etc/docker/daemon.json on your host:

{
  "insecure-registries": ["192.168.1.140:5000"]
}


Then restart Docker:

sudo systemctl restart docker

## Docker registry

docker run -d -p 5000:5000 --restart=always --name registry registry:2


## Start a cluster for mychatbot app

minikube start --cpus=16 --memory=8192 --driver=docker --insecure-registry "192.168.1.140:5000"
minikube status
minikube delete

## Test registry

curl -v http://192.168.1.140:5000/v2/
minikube ssh "curl -v http://192.168.1.140:5000/v2/"

## run act locally

act workflow_dispatch -W .github/workflows/deploy.yml --secret-file .secrets -s CLOUD=local
