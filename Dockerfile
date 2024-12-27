FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y bash

WORKDIR /app

RUN useradd -m -s /bin/bash llm

COPY . .

RUN chown -R llm:llm /app

USER llm
ENV PATH="/home/llm/.local/bin:${PATH}"
RUN pip install --user -r requirements-docker.txt
RUN pip uninstall -y numpy 
RUN pip install "numpy<2.0"
