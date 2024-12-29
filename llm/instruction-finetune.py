import time
import torch

from instruction.instruction_data import enumerate_test_data, load_instruction_data, format_input_to_alpaca, partition_data, InstructionDataset, custom_collate_fn
from instruction.dataloader import create_dataloader
from instruction.evaluate import is_ollama_running, generate_model_scores
from training import train_model_simple
from tokens import text_to_token_ids, token_ids_to_text
from model import get_device, get_model_path, get_tokenizer, save_model_dict
from functools import partial
from gpt_download import download_and_load_gpt2
from gpt import GPTModel, generate
from load_weights import load_weights_into_gpt
from model import get_base_config, get_model_size

# General settings
prefer_gpu = False

def instruction_fine_tune(model, tokenizer, train_loader, val_loader, device, val_data):    
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input_to_alpaca(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
def test_evaluation(model, tokenizer, device, test_data):
    BASE_CONFIG = get_base_config()
    for entry in test_data[:3]:
        input_text = format_input_to_alpaca(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        print(input_text)
        # print the model responses alongside the expected test set answers for the first three test set entries, presenting them side by side for comparison:
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

def main():
    tokenizer = get_tokenizer()
    
    # 1. Dataset download and formatting
    file_path = "data/instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = load_instruction_data(file_path, url)

    # Partition dataset in training, validation and test
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    
    train_data, test_data, val_data = partition_data(data, train_portion, test_portion, val_portion)

    # 2. Batching dataset

    # Use custom collate function to handle specific requirements and formatting of
    # instruction fine-tuning dataset.
    
    # Use partial to create custom_collate_fn with specific parameters    
    device = get_device(prefer_gpu)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )
    
    # 7.4. Setup data loaders
    num_workers = 0
    batch_size = 8

    train_loader = create_dataloader(train_data, tokenizer, batch_size, customized_collate_fn, num_workers)
    val_loader = create_dataloader(val_data, tokenizer, batch_size, customized_collate_fn, num_workers)
    test_loader = create_dataloader(test_data, tokenizer, batch_size, customized_collate_fn, num_workers)
    ## Examine dimensions of input and target batches
    # This output shows that the first input and target batch have dimensions 8 × 61, where 8 represents the batch size and 61 is the number of tokens in each training example in this batch
    #print("Train loader:")
    #for inputs, targets in train_loader:
    #    print(inputs.shape, targets.shape)
    
    # 7.5. Load pretrained LLM see fig. 7.15
    ## Load gpt2-medium (355M) params
    BASE_CONFIG = get_base_config()

    settings, params = download_and_load_gpt2(
        model_size=get_model_size(), 
        models_dir="data/gpt2"
    )
    
    model = GPTModel(BASE_CONFIG)
    # Load downloaded gpt2-medium (355M) params into model
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)
    
    # 7.6. fine-tune LLM on instruction data
    # Instruction fine-tune the pretrained LLM
    ## Calculate the initial loss for training & validation sets

    #torch.manual_seed(123)

    #with torch.no_grad():
    #    train_loss = calc_loss_loader(
    #        train_loader, model, device, num_batches=5
    #    )
    #    val_loss = calc_loss_loader(
    #        val_loader, model, device, num_batches=5
    #)

    #print("Initial Training loss:", train_loss)
    #print("Initial Validation loss:", val_loss)
    instruction_fine_tune(model, tokenizer, train_loader, val_loader, device, val_data)
    
    # 7.7 Extracting and saving responses, fig. 7.18
    ## Extract responses, evaluate LLM to quantify quality of responses
    test_evaluation(model, tokenizer, device, test_data)
        
    ## Save model for future usage
    save_model_dict(model, get_model_path(prefer_gpu))
    print("Model saved.")
    print("Next steps: Evaluate model responses using ollama.")

if __name__ == '__main__':
    main()
