FROM python:3.11-slim

WORKDIR /app

# System deps for scientific-Python wheels build faster with these present,
# even though manylinux wheels usually avoid needing a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

# GROQ_API_KEY / OPENROUTER_API_KEY are read from the environment at
# runtime (see .env.example) — pass them with `docker run -e ...` or
# `--env-file .env`, they are never baked into the image.

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
