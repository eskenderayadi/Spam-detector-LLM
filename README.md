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

## Local Setup

Use Python 3.10 or 3.11 and `uv` commands.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

## Recreate the Fine-Tuned Checkpoint

Install the development dependencies:

```bash
uv pip install -r requirements-dev.txt
```

Run `main.ipynb` from top to bottom. The notebook downloads GPT-2 weights, trains the LoRA classifier, evaluates it on the spam dataset, and saves it as `GPT2-with-LoRA-ft.pth`

**Note**: GPU acceleration is strongly recommended for training (e.g. Google Colab using "GPU-T4" runtime available as of the writing (May, 2026).

Finally, run the Chainlit app:

```bash
chainlit run app.py
```

Chainlit will automatically open the local URL. If it doesn’t, go to `http://localhost:8000`.

You can now send any message you’ve received if you have any doubts about whether it is spam or not.


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
