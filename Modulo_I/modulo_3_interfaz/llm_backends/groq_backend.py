"""
Backend Groq (Llama 3 hospedado). Activar:

    pip install groq
    export GROQ_API_KEY=gsk_...
"""
import os
import time

from .base import LLMBackend, LLMResponse, build_prompt


class GroqBackend(LLMBackend):
    name = "groq"
    model_id = "llama-3.3-70b-versatile"

    def __init__(self, model_id: str | None = None):
        if model_id:
            self.model_id = model_id
        try:
            from groq import Groq
        except ImportError as e:
            raise RuntimeError(
                "groq SDK no instalado. Ejecuta: pip install groq"
            ) from e
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY no esta exportada.")
        self.client = Groq(api_key=api_key)

    def generate(self, signal_json: dict) -> LLMResponse:
        prompt = build_prompt(signal_json)
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=600,
            )
            text = resp.choices[0].message.content
            return LLMResponse(
                text=text.strip(),
                backend_name=self.name,
                model_id=self.model_id,
                latency_seconds=time.time() - t0,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        except Exception as e:
            return LLMResponse(
                text="", backend_name=self.name, model_id=self.model_id,
                latency_seconds=time.time() - t0, error=str(e),
            )
