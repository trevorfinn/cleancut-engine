# CleanCut engine - ffmpeg + yt-dlp API; speech via the Groq API (GROQ_API_KEY).
# Deploys to any Docker host (built for Render's free plan).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clean.py server.py index.html wordlist.json ./

# Writable scratch for job files; the host injects PORT at runtime.
ENV CLEANCUT_JOBS=/tmp/cleancut-jobs
EXPOSE 8477

CMD ["python", "server.py"]
