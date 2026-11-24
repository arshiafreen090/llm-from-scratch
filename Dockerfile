# The chat UI + OpenAI-compatible API in one container, CPU only.
#
#   docker build -t tinyllama .
#   docker run -p 8000:7860 tinyllama                                   # open http://localhost:8000
#   docker run -p 8000:7860 -e TINYLLAMA_API_KEYS=sk-tl-... tinyllama   # require API keys
#
# The model (11 MB, int8) is downloaded from the GitHub release and checked against its SHA-256.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

ARG MODELS=https://github.com/arshiafreen090/llm-from-scratch/releases/download/models-v1
RUN mkdir -p checkpoints \
 && curl -fsSL -o checkpoints/sft-small-llama-s1-int8.pt $MODELS/sft-small-llama-s1-int8.pt \
 && echo "a3691f80b427f7fb79a0ebe9c7db89c23c16bb14f8145db88d325b8cb4adc5a5  checkpoints/sft-small-llama-s1-int8.pt" \
    | sha256sum -c -

# only the code the server needs
COPY tokenizer/ tokenizer/
COPY model/ model/
COPY inference/ inference/
COPY serve/ serve/

# run as a normal user, not root
RUN useradd -m -u 1000 app && chown -R app /app
USER app

# PyTorch can't always see a container's CPU limit and may start one thread per HOST core,
# which is much slower than matching the CPUs it really has. Set this to your CPU count.
ENV OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-m", "serve.api", "--ckpt", "checkpoints/sft-small-llama-s1-int8.pt", \
     "--host", "0.0.0.0", "--port", "7860", "--trust_proxy"]
