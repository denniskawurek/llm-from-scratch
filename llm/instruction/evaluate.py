import json
import psutil
from tqdm import tqdm
import urllib

from instruction.instruction_data import format_input_to_alpaca

def is_ollama_running():
    running = False
    for proc in psutil.process_iter(["name"]):
        if "ollama" in proc.info["name"]:
            running = True
            break
    return running

def query_model(
    prompt, 
    model="llama3", 
    url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            print(response_json)
            if "message" in response_json:
                if "content" in response_json["message"]:
                    response_content = response_json["message"]["content"]
                    try:
                        int(response_content)
                        response_data += response_content
                    except ValueError:
                        continue

    return response_data

## Run for whole dataset and evaluate
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input_to_alpaca(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores