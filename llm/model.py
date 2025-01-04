import os
import torch
import tiktoken

from gpt import GPTModel, generate
from tokens import text_to_token_ids, token_ids_to_text

def init_model(prefer_gpu):
    BASE_CONFIG = get_base_config()
    model = GPTModel(BASE_CONFIG)

    model_path = get_model_path(prefer_gpu)
    model.load_state_dict(load_model_dict(model_path))
    if prefer_gpu:
        model = model.to('cuda')

    return model

def generate_answer(input_text, model, device, config):
    print("Generating response...")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, get_tokenizer()).to(device),
        max_new_tokens=256,
        context_size=config["context_length"],
        eos_id=get_padding_token_id()
    )
    print("Generating response done...")

    generated_text = token_ids_to_text(token_ids, get_tokenizer())

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    return response_text

def load_model_dict(model_path):
    if os.path.exists(model_path) is False:
        raise RuntimeError("Model does not exist. Train the model first. See README.md for instructions.")

    model = torch.load(model_path, weights_only=True)
    print("Model loaded successfully.")
    return model

def save_model_dict(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

def get_model_path(prefer_gpu=True):
    if prefer_gpu:
        return "models/llm-cuda-sft.pth"
    return "models/llm-cpu-sft.pth"

def get_device(prefer_gpu=True):
    device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

'''
The model is a GPT model trained on gpt2-medium (355M)
'''
def get_base_config():
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    }

def model_name():
    return "gpt2-medium (355M)"
    
def get_model_size():
    return model_name().split(" ")[-1].lstrip("(").rstrip(")")

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

# => This returns the token_id for the padding token = 50256
def get_padding_token_id():
    return get_tokenizer().encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]