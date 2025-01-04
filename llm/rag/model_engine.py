from typing import Any
from llama_cloud import MessageRole
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from model import generate_answer, init_model, get_device, get_base_config
from gpt import GPTModel

import torch

from llama_index.core.bridge.pydantic import Field

class ModelEngine(CustomLLM):
    model: GPTModel = Field(default=None, description="The model to use.")
    device: torch.device = Field(default=None, description="The device.")
    config: dict[str, Any] = Field(default=None, description="The configuration of the model.")

    def __init__(self, prefer_gpu=True):
        super().__init__()
        self.model = init_model(prefer_gpu)
        self.device = get_device(prefer_gpu)
        self.config = get_base_config()
    
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=generate_answer(prompt, self.model, self.device, self.config),
            message_role=MessageRole.ASSISTANT
        )
        
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self._complete(prompt, formatted, **kwargs)
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window = self.config["context_length"],
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="ModelEngine",
            system_role=MessageRole.ASSISTANT
        )