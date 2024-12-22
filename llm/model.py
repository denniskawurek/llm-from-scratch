import os
import torch
import tiktoken

def load_model_dict(model_path):
    if os.path.exists(model_path) is False:
        print("Model does not exist. Train the model first. See README.md for instructions.")
        exit

    model = torch.load(model_path)
    print("Model loaded successfully.")
    return model

def save_model_dict(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

def get_model_path(prefer_gpu=True):
    if prefer_gpu:
        return "models/llm-gpu-sft.pth"
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