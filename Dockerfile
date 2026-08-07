# Build the maintained yt-dlp Proof-of-Origin token provider.
FROM node:22-bookworm-slim AS pot-builder
ARG BGUTIL_VERSION=1.3.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3 make g++ \
        libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${BGUTIL_VERSION}" \
        https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git \
        /opt/bgutil-ytdlp-pot-provider
WORKDIR /opt/bgutil-ytdlp-pot-provider/server
RUN npm ci && npx tsc

# CleanCut engine - ffmpeg + yt-dlp API; speech via the Groq API.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates libstdc++6 libcairo2 libpango-1.0-0 \
        libjpeg62-turbo libgif7 librsvg2-2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=pot-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=pot-builder /opt/bgutil-ytdlp-pot-provider \
    /opt/bgutil-ytdlp-pot-provider

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY clean.py server.py index.html wordlist.json ./

ENV CLEANCUT_JOBS=/tmp/cleancut-jobs
EXPOSE 8477
CMD ["python", "server.py"]
