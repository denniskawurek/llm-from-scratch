import json
import os
import urllib.request
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from model import get_base_config, get_padding_token_id
from gpt import generate
from tokens import text_to_token_ids, token_ids_to_text

def load_instruction_data(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# Converts the data input to Alpaca style
def format_input_to_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

# Partition dataset

def partition_data(data, train_portion, test_portion, val_portion):
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    return train_data, test_data, val_data


# TODO: Add this to own file
# fig. 7.6. steps 2.1 and 2.2
# Apply prompt template from above, use tokenization
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input_to_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

'''
fig. 7.6. step 2.3 add padding token. Append directly with token ID
fig. 7.6. step 2.5 replacing -100 placeholder tokens to mask padding tokens in the loss function.
Padding happens to have all inputs the same length. Apply to max length
Assign a -100 placeholder value to all padding tokens, as highlighted in figure 7.11. This special value allows us to exclude these padding tokens from contributing to the training loss calculation, ensuring that only meaningful data influences model learning. 
one end of text token is retained! See fig. 7.12

-100 is ignored by the cross entropy loss:
    The default setting of the cross entropy function in PyTorch is cross_entropy(..., ignore_index=-100)

It's also an option to mask out instruction tokens. See fig. 7.13

Verify by:

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
'''

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu" # "cpu", "cuda", "mps"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]


        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def enumerate_test_data(test_data, model, tokenizer, device):
    BASE_CONFIG = get_base_config()
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input_to_alpaca(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=get_padding_token_id()
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    with open("data/instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)
        
def get_enumerated_test_data_path():
    return "data/instruction-data-with-response.json"

def enumerated_test_data_exists():
    return os.path.exists(get_enumerated_test_data_path())

def load_enumerated_instruction_data():
    with open(get_enumerated_test_data_path(), "r") as file:
        data = json.load(file)
    return data