# Spam Detector LLM

A Chainlit app that classifies text messages as `spam` or `not spam` using a GPT-2 124M-style classifier fine-tuned with LoRA.

The model architecture and training workflow are based on Sebastian Raschka's book and reference implementation. See [Citation](#citation).

## Repository Contents

- `app.py` - Chainlit entry point for inference.
- `functions.py` - GPT model, LoRA layers, training helpers, and classifier utilities adapted from the book reference code.
- `main.ipynb` - notebook used to fine-tune the GPT-2 classifier and save `GPT2-with-LoRA-ft.pth`.
- `train.csv`, `validation.csv`, `test.csv` - SMS spam classification splits.
- `gpt_download.py` - helper for downloading GPT-2 weights used during training.
- `requirements.txt` - runtime dependencies for deploying the Chainlit app.
- `requirements-dev.txt` - optional notebook and training dependencies.
- `Dockerfile` - container entry point for hosted deployment.

## Model Files

The trained checkpoint `GPT2-with-LoRA-ft.pth` is about 533 MB, and the raw GPT-2 checkpoint under `gpt2/124M/` is also large. These files are intentionally excluded from normal Git commits because GitHub blocks files over 100 MB.

For deployment, provide the fine-tuned checkpoint using one of these options:

1. Put `GPT2-with-LoRA-ft.pth` in the project root before running the app.
2. Set `MODEL_PATH` to a checkpoint path.
3. Upload the checkpoint to durable storage and set `MODEL_URL` so the app can download it at startup.

Examples:

```bash
MODEL_PATH=/models/GPT2-with-LoRA-ft.pth chainlit run app.py
```

```bash
MODEL_URL=https://example.com/GPT2-with-LoRA-ft.pth chainlit run app.py
```

If you want the checkpoint in GitHub, use Git LFS:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add -f GPT2-with-LoRA-ft.pth
```

## Local Setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
uv -m pip install -r requirements.txt
```

Make sure the fine-tuned model is available:

```bash
ls GPT2-with-LoRA-ft.pth
```

Run the Chainlit app:

```bash
chainlit run app.py
```

Open the local URL shown by Chainlit, usually `http://localhost:8000`.

## Recreate the Fine-Tuned Checkpoint

Install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run `main.ipynb` from top to bottom. The notebook downloads GPT-2 weights, trains the LoRA classifier, evaluates it on the spam dataset, and saves:

```text
GPT2-with-LoRA-ft.pth
```

GPU acceleration is strongly recommended for training.

## Deploy With Docker

Build the image:

```bash
docker build -t spam-detector-llm .
```

Run with a mounted local checkpoint:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/models/GPT2-with-LoRA-ft.pth \
  -v "$PWD/GPT2-with-LoRA-ft.pth:/models/GPT2-with-LoRA-ft.pth:ro" \
  spam-detector-llm
```

Run with a hosted checkpoint:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_URL=https://example.com/GPT2-with-LoRA-ft.pth \
  spam-detector-llm
```

## Deploy On A Hosted Platform

Most Python container platforms work with this repository. Use these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `chainlit run app.py --host 0.0.0.0 --port $PORT`
- Environment variable: `MODEL_URL` or `MODEL_PATH`
- Python version: `3.11`

For Hugging Face Spaces, create a Docker Space and push this repository. Add `MODEL_URL` as a Space secret, or upload the checkpoint to the Space storage and set `MODEL_PATH`.

For Render, Railway, Fly.io, or similar platforms, deploy from the Dockerfile and configure `MODEL_URL` as an environment variable.

## Citation

This project uses code and ideas from:

Raschka, Sebastian. *Build a Large Language Model (From Scratch).* Manning Publications, 2024. ISBN: 978-1633437166. Book page: <https://www.manning.com/books/build-a-large-language-model-from-scratch>. Reference code: <https://github.com/rasbt/LLMs-from-scratch>.

The original reference code is published by Sebastian Raschka under the Apache License 2.0.
