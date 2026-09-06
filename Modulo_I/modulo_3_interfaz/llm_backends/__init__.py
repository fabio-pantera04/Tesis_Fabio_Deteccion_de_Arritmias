"""
Factory que escoge el backend segun configuracion. Permite ejecutar
las 5 senales contra cada backend para comparacion BLEU/ROUGE/BERTScore.
"""
from .base import LLMBackend, LLMResponse, build_prompt
from .mock_backend import MockBackend


_BACKENDS = {}


def get_backend(name: str) -> LLMBackend:
    """Devuelve una instancia (singleton) del backend pedido."""
    if name in _BACKENDS:
        return _BACKENDS[name]

    if name == "mock":
        backend = MockBackend()
    elif name == "claude":
        from .claude_backend import ClaudeBackend
        backend = ClaudeBackend()
    elif name == "gemini":
        from .gemini_backend import GeminiBackend
        backend = GeminiBackend()
    elif name == "groq":
        from .groq_backend import GroqBackend
        backend = GroqBackend()
    else:
        raise ValueError(
            f"Backend '{name}' no reconocido. "
            f"Opciones: mock, claude, gemini, groq."
        )

    _BACKENDS[name] = backend
    return backend


__all__ = ["LLMBackend", "LLMResponse", "build_prompt", "get_backend"]
