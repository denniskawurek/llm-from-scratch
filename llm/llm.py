import argparse
from evaluation import evaluate_model
from model import load_model_dict, get_device, get_base_config, get_model_path, get_tokenizer, get_padding_token_id
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the LLM model.')
    parser.add_argument('--prefer-gpu', action='store_true', help='Whether to prefer GPU')
    parser.add_argument('input', nargs='?', help='Instruction text for the model')

    args = parser.parse_args()
    prefer_gpu = args.prefer_gpu
    input_text = args.input

    model = init_model(prefer_gpu)
    device = get_device(prefer_gpu)
    
    if input_text:
        response_text = generate_answer(input_text, model, device, get_base_config())
        print(f"\nModel response:\n>> {response_text.strip()}")
    else: # Interactive mode
        while True:
            input_text = input("Enter your question (or type 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break
            response_text = generate_answer(input_text, model, device, get_base_config())
            print(f"\nModel response:\n>> {response_text.strip()}")

    #evaluate_model(model, get_tokenizer(), device)