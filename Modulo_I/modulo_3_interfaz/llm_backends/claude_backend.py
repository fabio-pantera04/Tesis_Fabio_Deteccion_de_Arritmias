"""
Backend Anthropic Claude. Activar instalando 'anthropic' y exportando
ANTHROPIC_API_KEY en el entorno.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import time

from .base import LLMBackend, LLMResponse, build_prompt


class ClaudeBackend(LLMBackend):
    name = "claude"
    model_id = "claude-sonnet-4-6"

    def __init__(self, model_id: str | None = None):
        if model_id:
            self.model_id = model_id
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK no instalado. Ejecuta: pip install anthropic"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no esta exportada.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, signal_json: dict) -> LLMResponse:
        prompt = build_prompt(signal_json)
        t0 = time.time()
        try:
            msg = self.client.messages.create(
                model=self.model_id,
                max_tokens=600,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            return LLMResponse(
                text=text.strip(),
                backend_name=self.name,
                model_id=self.model_id,
                latency_seconds=time.time() - t0,
                prompt_tokens=msg.usage.input_tokens,
                completion_tokens=msg.usage.output_tokens,
            )
        except Exception as e:
            return LLMResponse(
                text="", backend_name=self.name, model_id=self.model_id,
                latency_seconds=time.time() - t0, error=str(e),
            )
