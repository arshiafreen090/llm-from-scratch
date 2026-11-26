# The chat UI + OpenAI-compatible API in one container, CPU only.
#
#   docker build -t tinyllama .
#   docker run -p 8000:7860 tinyllama                                   # open http://localhost:8000
#   docker run -p 8000:7860 -e TINYLLAMA_API_KEYS=sk-tl-... tinyllama   # require API keys
#
# The model (11 MB, int8) is downloaded from the Hugging Face model repo and checked against its SHA-256.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

ARG MODELS=https://huggingface.co/afeening/tinyllama-from-scratch/resolve/main
ARG MODEL=sft-small-llama-s1-v2-int8
RUN mkdir -p checkpoints \
 && curl -fsSL -o checkpoints/$MODEL.pt $MODELS/$MODEL.pt \
 && echo "6ed2718aa3f50e47d245d88c07f67aee61bf2c2250e5b5feef780e25f761fb3f  checkpoints/$MODEL.pt" \
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
CMD ["python", "-m", "serve.api", "--ckpt", "checkpoints/sft-small-llama-s1-v2-int8.pt", \
     "--host", "0.0.0.0", "--port", "7860", "--trust_proxy"]
