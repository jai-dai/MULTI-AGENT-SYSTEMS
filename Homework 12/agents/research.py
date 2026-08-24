"""Researcher — исполняет план и докладывает, что нашёл.

Отчёт он НЕ пишет: итог собирает супервизор. Разделение не косметическое —
исследователь, который сам пишет итог, начинает подгонять находки под уже
написанный текст.

Единственный из троих, кому дан `read_url`: находки должны опираться на полный
текст источника, а не на сниппет выдачи.

Свободный текст вместо `response_format` здесь осознан. Находки — это проза с
цитатами и ссылками, и загонять её в модель данных значит либо потерять
структуру источников, либо изобретать схему под каждый тип запроса.

Промпт больше не лежит рядом с кодом: он приезжает из Langfuse по имени
`NAME` и лейблу из настроек. Роль агента теперь описана в ДВУХ местах —
набор инструментов здесь, текст задачи там, — и это осознанный размен:
промпт стало можно править и откатывать, не трогая деплой.
"""
from __future__ import annotations

from langchain.agents import create_agent

import observability

NAME = "researcher"
TOOLS = ["web_search", "read_url", "knowledge_search"]
DESCRIPTION = ("Executes a research plan against the knowledge base and the web, "
               "and reports findings with citations.")
SKILL_EXAMPLES = [
    "GOAL: compare naive and sentence-window retrieval. SEARCH QUERIES: ...",
]


def build(tools, model: str):
    return create_agent(
        model=model,
        tools=[t for t in tools if t.name in TOOLS],
        system_prompt=observability.prompt(NAME),
    )
