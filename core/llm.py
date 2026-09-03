"""
Thin wrapper around a local Ollama model.

Why this exists as its own module: every other component (lesson planner,
orchestrator, assessment, visuals) needs to call the LLM. Centralizing it
means we only have one place to change models, retry logic, or add
streaming later.
"""
from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class LLMConfig:
    model: str = "qwen2.5:7b"       # good balance of speed + multilingual + instruction following
    temperature: float = 0.4
    num_ctx: int = 8192              # bump if you have RAM headroom; needed for long lesson context


class LocalLLM:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def _call(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Could not reach Ollama at localhost:11434. "
                "Run `ollama serve` and make sure the model is pulled "
                f"(`ollama pull {self.config.model}`)."
            ) from e

    def generate(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        return self._call(prompt, system=system, temperature=temperature)

    def generate_json(self, prompt: str, system: str | None = None, temperature: float | None = None) -> dict:
        """
        Ask for JSON, then defensively extract it even if the model wraps it
        in markdown fences or adds a sentence before/after (small local models
        do this often, so don't assume clean output).
        """
        json_instruction = (
            "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, "
            "no explanation before or after the JSON object."
        )
        raw = self._call(prompt + json_instruction, system=system, temperature=temperature)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        raw = raw.strip()
        # strip markdown fences if present
        raw = re.sub(r"^```(json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # fallback: grab the largest {...} span
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model did not return parseable JSON:\n{raw[:500]}")
