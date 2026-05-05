# Raschka, Sebastian. Build A Large Language Model (From Scratch).
#   Manning, 2024. ISBN: 978-1633437166.

from pathlib import Path
import os
import sys
from urllib.request import urlretrieve

import tiktoken
import torch
import chainlit

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from functions import (
    GPTModel,
    replace_linear_with_lora,
    classify_review
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_path():
    model_path = Path(os.getenv("MODEL_PATH", "GPT2-with-LoRA-ft.pth"))
    if model_path.exists():
        return model_path

    model_url = os.getenv("MODEL_URL")
    if model_url:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model checkpoint to {model_path}...")
        urlretrieve(model_url, model_path)
        return model_path

    print(
        f"Could not find the {model_path} file. Set MODEL_PATH to an existing "
        "checkpoint, set MODEL_URL to download one at startup, or run "
        "main.ipynb to generate GPT2-with-LoRA-ft.pth."
    )
    sys.exit(1)


def get_model_and_tokenizer():
    """
    Code to load finetuned GPT-2 model generated in main.ipynb.
    This requires that you run the code in main.ipynb first, which generates and saves the necessary model.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = resolve_model_path()

    # Instantiate model
    model = GPTModel(GPT_CONFIG_124M)

    # Convert model to classifier
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)
    replace_linear_with_lora(model, rank=16, alpha=16)

    # Then load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # This returns the model response to the interface
    ).send()
