# Spam Detector LLM

A Chainlit app that classifies text messages as `spam` or `not spam` using OpenAI's GPT-2 124M-style classifier fine-tuned with  State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) **LoRA** (Low-Rank Adaptation).

The model architecture and training workflow are based on Sebastian Raschka's [book](https://www.manning.com/books/build-a-large-language-model-from-scratch) and [reference implementation](https://github.com/rasbt/LLMs-from-scratch). See [Citation](#citation).

**LoRA** is a parameter-efficient fine-tuning method proposed by Hu et al. (2021) that reduces the size of subsequent checkpoints. See [Citation](#citation). You may refer to the official Microsoft repository at [microsoft/LoRA](https://github.com/microsoft/LoRA) for further details and implementation examples.

## Repository Contents

- `app.py` - Chainlit entry point for inference.
- `functions.py` - GPT model, LoRA layers, training helpers, and classifier utilities adapted from the book reference code.
- `main.ipynb` - notebook used to fine-tune the GPT-2 classifier and save `GPT2-with-LoRA-ft.pth`.
- `train.csv`, `validation.csv`, `test.csv` - SMS spam classification splits.
- `gpt_download.py` - helper for downloading GPT-2 weights used during training.
- `requirements.txt` - runtime dependencies for deploying the Chainlit app.
- `requirements-dev.txt` - optional notebook and training dependencies.

## Local Setup

Use Python 3.10 or 3.11 and `uv` commands.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

## Recreate the Fine-Tuned Checkpoint


Run `main.ipynb` from top to bottom. The notebook downloads GPT-2 weights, trains the LoRA classifier, evaluates it on the spam dataset, and saves it as `GPT2-with-LoRA-ft.pth`

**Note**: GPU acceleration is strongly recommended for training (e.g. Google Colab using "GPU-T4" runtime available as of the writing (May, 2026). Below, the tutorial on the screenshots:

- Open `main.ipynb` on Colab and drag and drop the highlighted files to Colab:
<img width="1007" height="336" alt="image" src="https://github.com/user-attachments/assets/a94bee03-2acd-4600-ba86-acc009bd2237" />

- Select any supported GPU runtime, wait a moment for its setting up and run all cells: 
<img width="413" height="426" alt="image" src="https://github.com/user-attachments/assets/0ea261ad-612a-4520-9c5d-5753da951e39" />

- Add the generated `GPT2-with-LoRA-ft.pth` file to your local cloned folder.

- Finally, run the Chainlit app:

```bash
chainlit run app.py
```

Chainlit will automatically open the local URL. If it doesn’t, go to `http://localhost:8000`.

You can now send any message you’ve received if you have any doubts about whether it is spam or not.

## Citation

- This project uses code and ideas from:

Raschka, Sebastian. *Build a Large Language Model (From Scratch).* Manning Publications, 2024. ISBN: 978-1633437166. Book page: <https://www.manning.com/books/build-a-large-language-model-from-scratch>. Reference code: <https://github.com/rasbt/LLMs-from-scratch>.

The original reference code is published by Sebastian Raschka under the Apache License 2.0.

- LoRA method from:

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. _LoRA: Low-Rank Adaptation of Large Language Models._ International Conference on Learning Representations, 2022. URL: <https://doi.org/10.48550/arXiv.2106.09685>.

## Built With

- Chainlit
- OpenAI API
- Python
