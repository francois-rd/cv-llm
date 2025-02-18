from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .base import LLM, LLMOutput, Nickname
from ..prompting import ClusterPrompt


@dataclass
class TransformersConfig:
    # RNG seed for replication of results.
    seed: int = 314159

    # Whether to split out the system instructions (for LLMs that support it).
    use_system_prompt: bool = False

    # Whether to use the transformers.Pipeline chat templating functionality.
    use_chat: bool = False

    # Model quantization options for bitsandbytes.
    quantization: Optional[str] = None

    # See transformers.AutoModelForCausalLM.from_pretrained for details.
    # NOTE: Skip 'quantization_config', which is handled specially.
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {"trust_remote_code": True},
    )

    # See transformers.Pipeline for details.
    # NOTE: Skip 'task', 'model', and 'torch_dtype', which are handled specially.
    pipeline_params: Dict[str, Any] = field(default_factory=dict)

    # See transformers.GenerationConfig for details.
    generation_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "return_full_text": False,
            "max_new_tokens": 5,
            "num_return_sequences": 1,
        },
    )


class TransformersLLM(LLM):
    """Interface for all LLMs using the HuggingFace transformers.Pipeline API."""

    def __init__(
        self,
        nickname: Nickname,
        transformers_cfg: TransformersConfig,
        *args,
        **kwargs,
    ):
        # Delayed imports.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
            set_seed,
        )

        # Basic initialization.
        set_seed(transformers_cfg.seed)
        super().__init__(nickname, *args, **kwargs)
        self.cfg = transformers_cfg

        # Quantization.
        model_params = {**self.cfg.model_params}  # Convert OmegaConf -> dict
        if self.cfg.quantization is not None:
            if self.cfg.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.cfg.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unsupported quantization: {self.cfg.quantization}")
            model_params.update({"quantization_config": bnb_config})

        # Pipeline initialization.
        self.llm = pipeline(
            task="text-generation",
            model=AutoModelForCausalLM.from_pretrained(nickname, **model_params),
            tokenizer=AutoTokenizer.from_pretrained(nickname),
            torch_dtype=torch.bfloat16,
            **self.cfg.pipeline_params,
        )

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        # Convert the ChatPromptTemplate into a usable message structure.
        messages = []
        for message in prompt.template.messages:
            if isinstance(message, SystemMessagePromptTemplate):
                role = "system"
            elif isinstance(message, HumanMessagePromptTemplate):
                role = "user"
            else:
                raise ValueError(f"Unsupported role template: {type(message)}")
            messages.append({"role": role, "content": message.prompt.template})
        text = "\n\n".join(message["content"] for message in messages)

        # Feed that message structure in the right format to the LLM.
        if self.cfg.use_chat:
            if self.cfg.use_system_prompt:
                prompt = messages
            else:
                prompt = [{"role": "user", "content": text}]
        else:
            prompt = text
        output = self.llm(prompt, **self.cfg.generation_params)
        generated_text = output[0]["generated_text"]
        return LLMOutput(generated_text, error_message=None)
