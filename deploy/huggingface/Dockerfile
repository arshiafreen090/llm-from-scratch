# Dockerfile for a Hugging Face Space (Docker SDK). Upload it to the Space together with
# the README.md next to it: the Space then builds the code from GitHub and the model from
# the GitHub release, and needs nothing else. See README "Deploy" for the steps.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# the code: change REF to a tag or commit to pin a version
ARG REPO=https://github.com/arshiafreen090/llm-from-scratch.git
ARG REF=main
RUN git clone --depth 1 --branch $REF $REPO . \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

# the model, checked against its SHA-256
ARG MODELS=https://github.com/arshiafreen090/llm-from-scratch/releases/download/models-v1
RUN mkdir -p checkpoints \
 && curl -fsSL -o checkpoints/sft-small-llama-s1-int8.pt $MODELS/sft-small-llama-s1-int8.pt \
 && echo "a3691f80b427f7fb79a0ebe9c7db89c23c16bb14f8145db88d325b8cb4adc5a5  checkpoints/sft-small-llama-s1-int8.pt" \
    | sha256sum -c - \
 && chown -R user /app
USER user

# free Spaces have 2 vCPUs; PyTorch would otherwise start one thread per host core
ENV OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 7860
# API keys: add a Space secret named TINYLLAMA_API_KEYS (comma-separated). Without it the API
# is open, rate-limited per visitor IP (--trust_proxy reads it from the Space's proxy).
CMD ["python", "-m", "serve.api", "--ckpt", "checkpoints/sft-small-llama-s1-int8.pt", \
     "--host", "0.0.0.0", "--port", "7860", "--trust_proxy", "--rate_limit", "20"]
