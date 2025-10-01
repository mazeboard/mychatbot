# mychatbot

## run locally using docker compose

. scripts/load_env.sh
docker compose up --build

## run locally using kubernetes

./scripts/run_kube.sh

## add ui.mychatbot.com to /etc/hosts

echo "$(minikube ip) ui.mychatbot.com" | sudo tee -a /etc/hosts