"""
LLM client helpers for multiple providers:
- Grok (xAI)
- Gemini (Google)
- OpenRouter
- Ollama (local)
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any
import httpx


def chat_with_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    base_url: Optional[str] = None,
) -> str:
    """
    Unified chat interface.
    Returns the assistant's reply text.
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        return _ollama_chat(model, messages, temperature=temperature, max_tokens=max_tokens, base_url=base_url or "http://localhost:11434")

    if provider == "gemini":
        return _gemini_chat(model, messages, api_key=api_key, temperature=temperature, max_tokens=max_tokens)

    return _openai_compatible_chat(
        provider=provider,
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def _openai_compatible_chat(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
) -> str:
    from openai import OpenAI

    if provider == "grok":
        base_url = base_url or "https://api.x.ai/v1"
        if not api_key:
            raise ValueError("Grok (xAI) API key is required")
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        if not api_key:
            raise ValueError("OpenRouter API key is required")
    else:
        if not base_url:
            raise ValueError("base_url is required for generic OpenAI-compatible provider")

    client = OpenAI(api_key=api_key or "no-key", base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _gemini_chat(
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    import google.generativeai as genai

    if not api_key:
        raise ValueError("Gemini API key is required")

    genai.configure(api_key=api_key)

    system_instruction = None
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})

    model_name = model if model.startswith("models/") else f"models/{model}"
    generative_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )

    chat = generative_model.start_chat(history=contents[:-1] if len(contents) > 1 else [])
    last_user = contents[-1]["parts"][0] if contents else ""
    response = chat.send_message(
        last_user,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


def _ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.4,
    max_tokens: int = 4096,
    base_url: str = "http://localhost:11434",
) -> str:
    """Call local Ollama server."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


DEFAULT_MODELS = {
    "grok": "grok-2-latest",
    "gemini": "gemini-1.5-pro",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "ollama": "llama3.1",
}
