from instruction.evaluate import generate_model_scores, is_ollama_running
from instruction.instruction_data import enumerate_test_data, enumerated_test_data_exists, load_enumerated_instruction_data, load_instruction_data, partition_data

def prepare_data():
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
    
    return test_data


def evaluate_model(model, tokenizer, device):
    test_data = prepare_data()
    ## Prepare responses for evaluation process
    ## Append generated model responses to test_data dictionary
    ## And Save to instruction-data-with-response.json
    if not enumerated_test_data_exists():
        enumerate_test_data(test_data, model, tokenizer, device)
    else:
        test_data = load_enumerated_instruction_data()
    
    # 7.8 Evaluating the fine-tuned LLM
    ## Use ollama with llama3
    ## Download ollama and execute ollama run llama3
    ## Keep the session open or run ollama serve
    
    ollama_running = is_ollama_running()

    if not ollama_running:
        raise RuntimeError(
            "Ollama not running. Launch ollama before proceeding."
    )
    print("Ollama running:", ollama_running)

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


# Further improvements:
# - Adjust hyperparameters like learning rate, batch size, or epochs
# - Increase and diversify the training dataset
# - Experiment with different prompts or instruction formats
# - Use a larger pretrained model for better accuracy
