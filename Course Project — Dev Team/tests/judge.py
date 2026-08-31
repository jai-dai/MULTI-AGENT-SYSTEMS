"""Судья для тестов — через собственный слой, но ДРУГОЙ моделью.

Правило, выведенное в предыдущей работе и подтверждённое замером: судья обязан
быть другого вендора, чем система. Модель, оценивающая собственную работу,
оценивает её мягко — та же болезнь, против которой в самой команде поставлен QA.

Система здесь работает на `MODEL_NAME`, судья — на `JUDGE_MODEL_NAME`. Две
настройки, потому что это две разные роли, а не одна с оговоркой.
"""
from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.models.base_model import DeepEvalBaseLLM     # noqa: E402
from langchain.chat_models import init_chat_model          # noqa: E402


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    judge_model_name: str = "claude-opus-5"


judge_settings = JudgeSettings()


class ProjectJudge(DeepEvalBaseLLM):
    """Судья поверх LangChain, чтобы провайдер менялся строкой в .env."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or judge_settings.judge_model_name
        self.tokens = self.calls = 0
        super().__init__(model=self._model_name)

    def load_model(self):
        return init_chat_model(self._model_name)

    def get_model_name(self) -> str:
        return f"{self._model_name} (project judge)"

    def _run(self, prompt: str, schema: type[BaseModel] | None):
        model = self.load_model()
        self.calls += 1
        if schema is None:
            reply = model.invoke(prompt)
            self._count(reply)
            return reply.content
        structured = model.with_structured_output(schema)
        return structured.invoke(prompt)

    def _count(self, reply) -> None:
        usage = getattr(reply, "usage_metadata", None) or {}
        self.tokens += usage.get("total_tokens", 0)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self._run(prompt, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def spent(self) -> str:
        return f"{self.tokens} токенов за {self.calls} вызовов судьи"
