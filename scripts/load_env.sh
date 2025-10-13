#!/usr/bin/env bash
set -a  # automatically export all variables
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  source .env
fi
set +a

