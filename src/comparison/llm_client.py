"""
src/comparison/llm_client.py
======================================
LocalLLMClient — Client gọi Qwen2.5-7B-Instruct qua API tương thích OpenAI.

Hỗ trợ:
  - JSON mode (response_format={"type": "json_object"})
  - Async/Await thông qua openai.AsyncOpenAI
  - Retry với exponential backoff
  - Configurable timeout, temperature, max_tokens

Server backend có thể là:
  - llama-cpp-python (python -m llama_cpp.server --model ... --n_gpu_layers -1)
  - vLLM (vllm serve Qwen/Qwen2.5-7B-Instruct --api-key none)
  - Ollama (ollama serve → base_url=http://localhost:11434/v1)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Cấu hình kết nối và inference cho LocalLLMClient."""

    # Endpoint
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL của OpenAI-compatible server (không có trailing slash)",
    )
    api_key: str = Field(
        default="not-needed",
        description="API key placeholder (local server thường không cần)",
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Tên model — phải khớp với model được load trên server",
    )

    # Inference hyper-parameters
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description=(
            "Nhiệt độ sampling. THẤP (0.0–0.2) cho structured JSON output. "
            "Dùng 0.0 cho Zero-Hallucination mode."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Số token tối đa trong response",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Timeout mỗi request (giây)",
    )

    # Retry policy
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Số lần retry tối đa khi gặp lỗi kết nối / timeout",
    )
    retry_base_delay: float = Field(
        default=1.0,
        gt=0,
        description="Delay cơ bản giữa các lần retry (giây) — exponential backoff",
    )

    # JSON mode
    force_json_mode: bool = Field(
        default=True,
        description=(
            "Bật response_format={'type': 'json_object'} — "
            "yêu cầu server hỗ trợ JSON mode (vLLM, llama-cpp-python >= 0.2.x)"
        ),
    )


# ---------------------------------------------------------------------------
# LocalLLMClient
# ---------------------------------------------------------------------------


class LocalLLMClient:
    """
    Client async để gọi local LLM (Qwen2.5-7B-Instruct) qua OpenAI-compatible API.

    Pattern sử dụng:
        client = LocalLLMClient(config=LLMConfig(base_url="http://localhost:8000/v1"))

        # Gọi với JSON mode (cho ACU extraction)
        result_json = await client.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
        )

        # Gọi thông thường (cho Executive Summary)
        text = await client.chat_text(
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=SUMMARY_USER,
        )
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._client = AsyncOpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout_seconds,
        )
        logger.info(
            "LocalLLMClient khởi tạo: model=%s, base_url=%s",
            self._config.model_name,
            self._config.base_url,
        )

    @property
    def config(self) -> LLMConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Gọi LLM với JSON mode bật.

        Returns:
            dict được parse từ JSON response của LLM.

        Raises:
            ValueError: Nếu response không parse được thành JSON sau tất cả retries.
            RuntimeError: Nếu gặp lỗi kết nối sau tất cả retries.
        """
        raw = await self._chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            extra_params=extra_params,
        )
        return self._parse_json_response(raw)

    async def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Gọi LLM và trả về plain text response.

        Returns:
            Chuỗi text từ LLM (đã strip).
        """
        return await self._chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=False,
            extra_params=extra_params,
        )

    async def health_check(self) -> bool:
        """Kiểm tra kết nối đến LLM server."""
        try:
            models = await self._client.models.list()
            available = [m.id for m in models.data]
            logger.info("LLM server OK. Models available: %s", available)
            return True
        except Exception as exc:
            logger.error("LLM server health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Thực hiện API call với retry logic.

        Args:
            system_prompt: Nội dung system message.
            user_prompt:   Nội dung user message.
            json_mode:     Bật response_format=json_object nếu True.
            extra_params:  Override thêm params (temperature, max_tokens, ...).

        Returns:
            Raw content string từ LLM response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        call_params: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        if json_mode and self._config.force_json_mode:
            call_params["response_format"] = {"type": "json_object"}

        if extra_params:
            call_params.update(extra_params)

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = await self._client.chat.completions.create(**call_params)
                elapsed = time.monotonic() - t0

                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

                logger.debug(
                    "LLM call OK: attempt=%d, elapsed=%.2fs, finish_reason=%s, "
                    "prompt_tokens=%d, completion_tokens=%d",
                    attempt + 1,
                    elapsed,
                    finish_reason,
                    response.usage.prompt_tokens if response.usage else -1,
                    response.usage.completion_tokens if response.usage else -1,
                )

                if finish_reason == "length":
                    logger.warning(
                        "LLM response bị cắt ngắn (finish_reason=length). "
                        "Cân nhắc tăng max_tokens=%d.",
                        self._config.max_tokens,
                    )

                return content.strip()

            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._config.max_retries:
                    delay = self._config.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retry in %.1fs...",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM call failed sau %d attempts: %s",
                        self._config.max_retries + 1,
                        exc,
                    )

            except RateLimitError as exc:
                # Rate limit trên local server thường không xảy ra,
                # nhưng nếu dùng multiple workers thì có thể gặp
                last_exc = exc
                delay = self._config.retry_base_delay * (2 ** attempt) * 2
                logger.warning(
                    "RateLimitError (attempt %d/%d). Retry in %.1fs...",
                    attempt + 1,
                    self._config.max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"LocalLLMClient: Exhausted {self._config.max_retries + 1} attempts. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """
        Parse JSON từ LLM response.

        Xử lý các trường hợp LLM wrap JSON trong markdown code block:
            ```json\n{...}\n```
        """
        text = raw.strip()

        # Strip markdown code fences nếu có
        if text.startswith("```"):
            lines = text.splitlines()
            # Bỏ dòng đầu (```json hoặc ```) và dòng cuối (```)
            inner_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner_lines.append(line)
            text = "\n".join(inner_lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Thử tìm JSON object đầu tiên trong text (một số model thêm preamble)
            import re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise ValueError(
                f"LocalLLMClient: Không thể parse JSON từ response. "
                f"Error: {exc}. Raw (first 500 chars): {raw[:500]!r}"
            ) from exc
