# Stage 1: Build React app
FROM node:18 AS build
WORKDIR /usr/src/app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -m nltk.downloader punkt punkt_tab

ARG DEBUG=false
RUN if [ "$DEBUG" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    curl wget iputils-ping net-tools dnsutils lsof procps \
    && rm -rf /var/lib/apt/lists/*; \
    fi

COPY . .

# Just document the mount point (does not create the models)
VOLUME ["/models"]

ENV SEMANTIC_LLM_MODEL_PATH=${SEMANTIC_LLM_MODEL_PATH}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_PROJECT_ID=${OPENAI_PROJECT_ID}
ENV AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT}
ENV AZURE_STORAGE_KEY=${AZURE_STORAGE_KEY}
ENV AZURE_CONTAINER=${AZURE_CONTAINER}

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]
