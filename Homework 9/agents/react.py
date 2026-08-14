"""Один цикл ReAct на всех агентов системы.

Обобщён из `agent.py` в hw5 — там он был написан под единственного агента с
жёстко зашитым набором инструментов. Здесь параметризовано ровно то, чем
агенты отличаются друг от друга: промпт, набор инструментов, лимит шагов и
форма ответа. Четыре агента — четыре экземпляра, а не четыре цикла: иначе
починка «модель зациклилась на одном инструменте» делалась бы четыре раза.

# Что добавилось против hw5

`output_model` — агент обязан закончить вызовом `submit_*` с валидной моделью
данных (см. structured.py). Пока он этого не сделал, обычные инструменты ему
доступны: Planner сначала осматривает домен, Critic сначала проверяет факты, и
только потом сдают структуру. Это и есть «tools + response_format вместе».

`interceptor` — крючок ПЕРЕД исполнением инструмента. Через него сделан
human-in-the-loop: супервизор зовёт `save_report`, крючок возвращает решение
человека, и оно уходит модели как результат инструмента. Отдельный механизм
приостановки графа не нужен — цикл синхронный, и «спросить человека» здесь
буквально означает спросить человека.

`depth` — только для вывода. Вложенность видна отступом, иначе вызовы
суб-агента и вызовы супервизора сливаются в одну ленту.
"""
from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel

import structured
from config import settings
from llm import LLMError, get_backend

LOG_PREVIEW = 160


def _preview(text: str, limit: int = LOG_PREVIEW) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _format_args(raw: str) -> str:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return _preview(raw, 80)
    return ", ".join(f"{k}={_preview(json.dumps(v, ensure_ascii=False), 60)}"
                     for k, v in data.items())


class ReactAgent:
    """Агент с собственным промптом, инструментами и формой ответа."""

    def __init__(self, name: str, system_prompt: str,
                 registry: dict[str, Callable], schemas: list[dict],
                 max_steps: int | None = None,
                 output_model: type[BaseModel] | None = None,
                 depth: int = 0,
                 interceptor: Callable[[str, str], str | None] | None = None) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.registry = registry
        self.output_model = output_model
        self.depth = depth
        self.interceptor = interceptor
        self.max_steps = max_steps or settings.max_iterations
        self.llm = get_backend()
        self.schemas = list(schemas)
        if output_model is not None:
            self.schemas.append(structured.tool_schema(output_model))
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.tokens = 0
        self.calls = 0

    # ------------------------------------------------------------------ #

    @property
    def _pad(self) -> str:
        return "  " * self.depth

    def _say(self, line: str) -> None:
        print(f"{self._pad}{line}", flush=True)

    def run(self, request: str) -> str | BaseModel:
        """Одна задача до конца. Текст, либо экземпляр `output_model`."""
        self.messages.append({"role": "user", "content": request})
        self.tokens = self.calls = 0

        for step in range(1, self.max_steps + 1):
            # На последнем шаге структурированному агенту больше нельзя гулять по
            # инструментам: если он и сейчас не сдаст результат, вызывающая
            # сторона останется ни с чем. Поэтому вызов принуждается.
            force = (self.output_model is not None and step == self.max_steps)
            reply = self._call(require=structured.tool_name(self.output_model)
                               if force else None)
            self.tokens += reply.tokens
            self.messages.append(self.llm.assistant_entry(reply))

            if reply.text and reply.tool_calls:
                self._say(f"💭 {_preview(reply.text, 200)}")

            if not reply.tool_calls:
                if self.output_model is not None:
                    # Ответил текстом там, где ждали структуру. Не ошибка модели
                    # в строгом смысле — напоминаем и продолжаем.
                    self.messages.append({
                        "role": "user",
                        "content": (f"Do not answer in prose. Call "
                                    f"{structured.tool_name(self.output_model)} "
                                    "with the result."),
                    })
                    continue
                return reply.text or ""

            for call in reply.tool_calls:
                done, value = self._handle(call)
                if done:
                    return value
        return self._out_of_steps()

    # ------------------------------------------------------------------ #

    def _handle(self, call) -> tuple[bool, str | BaseModel]:
        """Исполнить один вызов. (True, результат) — если работа закончена."""
        self.calls += 1
        self._say(f"🔧 {call.name}({_format_args(call.arguments)})")

        if (self.output_model is not None
                and call.name == structured.tool_name(self.output_model)):
            value, error = structured.parse(self.output_model, call.arguments)
            if value is not None:
                self._say(f"📎 {self.output_model.__name__} готов")
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": "accepted"})
                return True, value
            self._say(f"⚠️  {_preview(error)}")
            self.messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": error})
            return False, ""

        result = self._dispatch(call.name, call.arguments)
        marker = "⚠️ " if result.startswith("ERROR") else ""
        self._say(f"📎 {marker}[{len(result)} chars] {_preview(result)}")
        self.messages.append({"role": "tool", "tool_call_id": call.id,
                              "content": result})
        return False, ""

    def _dispatch(self, name: str, raw_arguments: str) -> str:
        """Исполнение с теми же гарантиями, что и в hw5: ошибка — это текст.

        Любая беда внутри инструмента возвращается моделью читаемой строкой, а
        не исключением: цикл обязан пережить кривой аргумент и попробовать иначе.
        """
        if self.interceptor is not None:
            verdict = self.interceptor(name, raw_arguments)
            if verdict is not None:
                return verdict

        func = self.registry.get(name)
        if func is None:
            return (f"ERROR: unknown tool '{name}'. "
                    f"Available: {', '.join(self.registry)}.")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments for '{name}' are not valid JSON ({exc})."
        if not isinstance(arguments, dict):
            return f"ERROR: arguments for '{name}' must be a JSON object."
        try:
            return func(**arguments)
        except TypeError as exc:
            return f"ERROR: bad arguments for '{name}' ({exc})."
        except Exception as exc:
            return f"ERROR: tool '{name}' crashed ({type(exc).__name__}: {exc})."

    def _out_of_steps(self) -> str | BaseModel:
        """Шаги кончились. Структурному агенту это фатально, текстовому — нет."""
        if self.output_model is not None:
            raise RuntimeError(
                f"{self.name}: за {self.max_steps} шагов так и не сдал "
                f"{self.output_model.__name__}")
        self._say(f"⏹  лимит шагов ({self.max_steps}) — прошу итог без инструментов")
        self.messages.append({
            "role": "system",
            "content": ("Step budget exhausted. You no longer have tools. "
                        "Answer now with what you already have, and state "
                        "plainly what remained unverified."),
        })
        reply = self._call(with_tools=False)
        self.tokens += reply.tokens
        self.messages.append({"role": "assistant", "content": reply.text})
        return reply.text or ""

    def _call(self, *, with_tools: bool = True, require: str | None = None):
        try:
            return self.llm.complete(self.messages,
                                     self.schemas if with_tools else None,
                                     require_tool=require)
        except LLMError as exc:
            raise RuntimeError(f"{self.name}: model call failed: {exc}") from exc

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
