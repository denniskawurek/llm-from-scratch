# Implementing an LLM from Scratch

This implements the LLM as described in [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka.

The concepts and most of the code pieces are taken from the book. I did modifications to enhance the usability and have an llm which runs out of the box and uses all the described concepts in the book.

See also the [GitHub repository](https://github.com/rasbt/LLMs-from-scratch) for the original code of the book.

Most of the code is commented with few sentences what it is doing and where to look at in the book to find more information.

The intention of this repository is to exist for learning-purposes.

## Setup

This repository uses `poetry` for managing virtual environments and packages.

1. Clone the repository
```sh
git clone git@github.com:denniskawurek/llm-from-scratch.git
```
2. Install Poetry: [Installation Guide](https://python-poetry.org/docs/#installation)
3. Install dependencies:
```sh
poetry install
```
4. Start a shell:
```sh
poetry shell
```

Within the shell the following commands can be executed:

## Fine-tune LLM

Before the LLM can be used, the model needs to be fine-tuned.

The LLM fine-tunes a `gpt2-medium (355M)` and loads the weights to the `GPTModel` class in `gpt.py`.

Run `instruction-finetune.py` to kick-off the process:

```sh
python llm/instruction-finetune.py
```

This loads some data. After that the finetuning process starts which may take some time depending on the hardware.

The model is stored in the `models` directory.

## Evaluating the model

The fine-tuned model can be evaluated with ollama.

For this install `ollama` and run:

```sh
ollama serve
```

and then in a different window:

```sh
ollama run llama3
```

Now uncomment the following line in `llm.py` and run `python llm.py` afterwards:
```python
# evaluate_model(model, get_tokenizer(), device)
```

## Use and run the LLM

Use `llm.py` to run the LLM. Right now it contains the `input_text` variable. This is the instruction which can be set for the LLM.

The following command starts an interactive session where the model is loaded and can receive instructions:

```sh
python llm/llm.py
```

The following command loads just the model and processes the given instruction:

```sh
python llm/llm.py 'What is the capital of United Kingdom?'
```

### Options

`--prefer-gpu` - use GPU for generating responses.

## Using GPU/CPU

`instruction-finetune.py` has a `prefer_gpu` variable. If this is set to true, `cuda` will be used to train and run the model. Default is `cpu` (or `mps` for MacBooks).

Beware that only a model which was finetuned with `cuda` can be used in `llm.py` with `prefer_gpu=True`