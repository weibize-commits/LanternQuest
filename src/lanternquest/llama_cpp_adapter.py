from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from lanternquest.llm import LLMAdapterError, LLMRequest, LLMResponse

_DLL_HANDLES: list[Any] = []


def _enable_windows_runtime_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    library_root = site_packages / "llama_cpp" / "lib"
    if library_root.is_dir():
        _DLL_HANDLES.append(os.add_dll_directory(str(library_root)))


class LlamaCppAdapter:
    """Local structured-output adapter backed by llama-cpp-python."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str,
        model_sha256: str,
        n_ctx: int = 8192,
        n_batch: int = 256,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        max_output_tokens: int = 2000,
        temperature: float = 0.0,
        seed: int = 21021,
        verbose: bool = False,
        client: Any | None = None,
    ) -> None:
        if not model_path.is_file() and client is None:
            raise FileNotFoundError(model_path)
        if len(model_sha256) != 64:
            raise ValueError("model_sha256 must contain 64 hexadecimal characters")
        if max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")
        if client is None:
            _enable_windows_runtime_dlls()
            from llama_cpp import Llama

            client = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                seed=seed,
                flash_attn=True,
                verbose=verbose,
            )
        self.client = client
        self.model_id = model_id
        self.model_sha256 = model_sha256
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed

    def complete(self, request: LLMRequest) -> LLMResponse:
        instructions = (
            f"{request.system_instruction} Return exactly one JSON object. "
            "Do not output analysis, Markdown, or code fences."
        )
        payload = json.dumps(request.payload, ensure_ascii=False, sort_keys=True)
        try:
            response = self.client.create_chat_completion(
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": payload},
                ],
                response_format={
                    "type": "json_object",
                    "schema": request.response_schema,
                },
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_output_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            if not isinstance(output_text, str) or not output_text.strip():
                raise ValueError("local model returned empty structured output")
            content = json.loads(output_text)
            if not isinstance(content, dict):
                raise ValueError("local model JSON output is not an object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LLMAdapterError(
                str(error),
                provider_calls=1,
            ) from error
        usage = response.get("usage", {})
        return LLMResponse(
            model_id=str(response.get("model", self.model_id)),
            content=content,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            response_id=str(response.get("id", "")) or None,
            system_fingerprint=f"sha256:{self.model_sha256}",
        )
