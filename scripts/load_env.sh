#!/usr/bin/env bash
set -a  # automatically export all variables
if [ -f .env ]; then
  source .env
fi
set +a

