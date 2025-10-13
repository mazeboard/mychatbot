# mychatbot

## run locally using docker compose

. scripts/load_env.sh
docker compose up --build

## ingest document

curl -X POST "http://localhost:8000/ingest" -F "file=@/home/taoufik/Downloads/Profile.pdf"

## run locally using kubernetes

./scripts/run_kube.sh

## add ui.mychatbot.com to /etc/hosts

echo "$(minikube ip) ui.mychatbot.com" | sudo tee -a /etc/hosts