import argparse
from evaluation import evaluate_model
from model import generate_answer, get_device, get_base_config, init_model

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