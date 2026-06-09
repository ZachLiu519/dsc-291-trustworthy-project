from __future__ import annotations

from typing import Sequence

from vllm import LLM, SamplingParams

from jbb_repro.artifacts import AttackPrompt
from jbb_repro.config import GenerationConfig, ModelConfig
from jbb_repro.generation import GenerationResult, batched, format_vicuna_prompt


class LocalVLLMGenerator:
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig) -> None:
        kwargs = {
            "model": model_config.model_name,
            "dtype": model_config.dtype,
            "tensor_parallel_size": model_config.tensor_parallel_size,
            "gpu_memory_utilization": model_config.gpu_memory_utilization,
            "trust_remote_code": model_config.trust_remote_code,
        }
        if model_config.quantization:
            kwargs["quantization"] = model_config.quantization

        self.model_name = model_config.model_name
        self.batch_size = generation_config.batch_size
        self.llm = LLM(**kwargs)
        self.sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=generation_config.max_new_tokens,
        )

    def generate(self, prompts: Sequence[AttackPrompt]) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for batch in batched(list(prompts), self.batch_size):
            rendered_prompts = [format_vicuna_prompt(prompt.prompt) for prompt in batch]
            outputs = self.llm.generate(rendered_prompts, self.sampling_params)
            for attack_prompt, output in zip(batch, outputs, strict=True):
                results.append(GenerationResult(prompt=attack_prompt, response=output.outputs[0].text.strip()))
        return results
