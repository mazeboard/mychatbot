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

ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_PROJECT_ID=${OPENAI_PROJECT_ID}
ENV GOOGLE_SEARCH_CSE_ID=${GOOGLE_SEARCH_CSE_ID}
ENV GOOGLE_SEARCH_API_KEY=${GOOGLE_SEARCH_API_KEY}

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${AI_SERVICE_PORT:-8000}"]
