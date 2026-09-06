"""
Backend Google Gemini 2.5 Flash. Activar:

    pip install google-generativeai
    export GEMINI_API_KEY=AIza...
"""
import os
import time

from .base import LLMBackend, LLMResponse, build_prompt


class GeminiBackend(LLMBackend):
    name = "gemini"
    model_id = "gemini-2.5-flash"

    def __init__(self, model_id: str | None = None):
        if model_id:
            self.model_id = model_id
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai no instalado. "
                "Ejecuta: pip install google-generativeai"
            ) from e
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no esta exportada.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_id)

    def generate(self, signal_json: dict) -> LLMResponse:
        prompt = build_prompt(signal_json)
        t0 = time.time()
        try:
            import google.generativeai as genai
            cfg = genai.types.GenerationConfig(
                temperature=0.0, max_output_tokens=600,
            )
            resp = self.client.generate_content(prompt, generation_config=cfg)
            return LLMResponse(
                text=resp.text.strip(),
                backend_name=self.name,
                model_id=self.model_id,
                latency_seconds=time.time() - t0,
                prompt_tokens=getattr(resp.usage_metadata, "prompt_token_count", None),
                completion_tokens=getattr(resp.usage_metadata, "candidates_token_count", None),
            )
        except Exception as e:
            return LLMResponse(
                text="", backend_name=self.name, model_id=self.model_id,
                latency_seconds=time.time() - t0, error=str(e),
            )
