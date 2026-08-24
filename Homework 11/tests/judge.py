"""LLM-судья для DeepEval — поверх нашего же `llm.py`.

# Судья ОБЯЗАН быть другим вендором

Лекция 11 называет это антипаттерном прямо: «використання тієї ж моделі як судді
та як target system → self-enhancement bias». Модель, оценивающая собственную
работу, оценивает её мягко, и это та же болезнь, против которой в самой системе
поставлен критик (см. README hw9 про «невиправдану впевненість»). Тест,
унаследовавший болезнь тестируемого, не измеряет ничего.

Поэтому модель судьи задаётся ОТДЕЛЬНО, через `JUDGE_MODEL_NAME`, и по умолчанию
это модель другого вендора. `MODEL_NAME` при этом остаётся моделью системы —
две настройки, потому что это две разные роли, а не одна с оговоркой.

# Зачем обёртка, если у DeepEval есть свои модели

В задании метрики создаются как `GEval(..., model="gpt-5.4-mini")`, то есть
судья задаётся строкой и ходит к провайдеру мимо проекта. Для одной домашки это
короче, но здесь так нельзя по той же причине, по которой в hw8 не взят
LangChain: смысл `llm.py` в том, что провайдер переключается одной строкой в
`.env`. Судья, вбитый строкой в тест, эту строку игнорирует — и в системе,
собранной вокруг переключения провайдера, появляется место, где оно не работает.

Заметь, что одно другому не противоречит: судья ходит через `llm.py`, но с
другим именем модели. Ради этого в `llm.py` появился параметр `model` у
`complete()`, а `get_backend()` стал держать клиент на ПРОТОКОЛ, а не один на
процесс — два вендора теперь работают в одном процессе одновременно. Правка
сделана в Agent_1 (источник цепочки копий), а не здесь: иначе она бы уехала при
первой же синхронизации.

# Как получается структурированный ответ

DeepEval зовёт `generate(prompt, schema=SomeModel)` и ждёт назад экземпляр
модели данных. У нас ровно этот механизм уже написан — `structured.py`: схема
Pydantic превращается в инструмент `submit_*`, и модель обязывают его вызвать
(`require_tool`). Никакого `response_format` не нужно, и это опять то же
решение, что в агентах: инструменты есть у обоих протоколов, а JSON-режим — нет.

Валидация тоже повторяет агентскую: не разобралось — ошибка возвращается модели
её же текстом, и она переспрашивает. Судья, молча вернувший мусор, хуже судьи,
переспросившего один раз: оценка 0.0 из-за неразобранного JSON выглядит в
отчёте как «система плохая», а не как «метрика не сработала».

# Почему судья считает свои токены

Главный вывод hw9 — что мультиагент стоит денег и что деньги надо ВИДЕТЬ
(см. `supervisor.py`). Оценка их тоже стоит, и без счётчика разговор «дорого ли
тестировать» опять стал бы разговором ощущений. Счётчик глобальный, потому что
один судья обслуживает все метрики прогона.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

import llm
import structured
from deepeval.models.base_model import DeepEvalBaseLLM
from llm import LLMError


class JudgeSettings(BaseSettings):
    """Настройки ОЦЕНКИ, отдельно от настроек системы.

    Свой класс, а не поле в `config.Settings`, по той же механической причине,
    по которой у hw10 свой `requirements-eval.txt`: `config.py` приезжает из
    Agent_1 по цепочке копий и перезаписывается целиком. Поле, дописанное туда,
    исчезло бы при первой синхронизации — молча.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # Другой вендор, не gpt-*: см. шапку модуля. Меняется в .env одной строкой,
    # ровно как MODEL_NAME у системы.
    judge_model_name: str = "claude-opus-5"


judge_settings = JudgeSettings()

# Сколько раз переспросить судью, если он вернул не то. Один повтор ловит
# случайную кривизну; упорно кривой ответ — это уже сигнал, а не помеха.
MAX_RETRIES = 2


class ProjectJudge(DeepEvalBaseLLM):
    """Судья DeepEval, говорящий через `llm.py` — тем же провайдером, что агент."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or judge_settings.judge_model_name
        self.tokens = 0
        self.calls = 0
        super().__init__(model=self._model_name)

    # DeepEval требует load_model; у нас соединение заводит сам llm.py, и
    # заводит его ОДНО на процесс — судье не нужен свой клиент.
    def load_model(self) -> Any:
        return llm.get_backend(self._model_name)

    def get_model_name(self) -> str:
        return llm.describe(self._model_name)

    def same_vendor_as_system(self) -> bool:
        """Судья и система — один протокол? Тогда оценка завышена, и это надо знать."""
        return (llm.resolve_backend(self._model_name)
                == llm.resolve_backend())

    # -- синхронный путь --------------------------------------------------- #

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return self._plain(prompt)
        return self._structured(prompt, schema)

    def _plain(self, prompt: str) -> str:
        reply = self._complete([{"role": "user", "content": prompt}], None, None)
        return reply.text or ""

    def _structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        name = structured.tool_name(schema)
        messages = [{"role": "user", "content": prompt}]
        tools = [structured.tool_schema(schema)]

        last_error = "the judge returned no tool call at all"
        for _ in range(MAX_RETRIES + 1):
            reply = self._complete(messages, tools, name)
            call = next((c for c in reply.tool_calls if c.name == name), None)
            if call is None:
                # Ответил прозой там, где принуждали к инструменту. Бывает у
                # провайдеров, трактующих tool_choice как пожелание.
                messages.append({"role": "user",
                                 "content": f"Do not answer in prose. Call {name}."})
                continue
            value, error = structured.parse(schema, call.arguments)
            if value is not None:
                return value
            last_error = error
            messages.append(self.load_model().assistant_entry(reply))
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": error})

        raise LLMError(
            f"судья не смог вернуть {schema.__name__} за {MAX_RETRIES + 1} "
            f"попытки: {last_error}")

    def _complete(self, messages: list[dict], tools, require: str | None):
        reply = self.load_model().complete(messages, tools, require_tool=require,
                                           model=self._model_name)
        self.tokens += reply.tokens
        self.calls += 1
        return reply

    # -- асинхронный путь -------------------------------------------------- #
    #
    # DeepEval по умолчанию гоняет метрики параллельно и зовёт `a_generate`.
    # Наш бэкенд синхронный (и это осознанно — см. шапку `bridge.py` в hw9),
    # поэтому вызов уезжает в поток. Параллелизм при этом настоящий: время здесь
    # уходит в ожидание сети, а не в GIL.

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    # -- отчёт ------------------------------------------------------------- #

    def spent(self) -> str:
        return f"{self.tokens} токенов за {self.calls} вызовов судьи"
