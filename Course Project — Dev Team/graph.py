"""Граф команды: BA → человек → Developer ⇄ QA.

# Почему именно граф, а не цикл

В предыдущих работах координатор был агентом: он сам решал, кого позвать, и
маршрут жил в промпте. Здесь маршрут — это РЁБРА, и разница видна на одном
месте. Возврат от QA к Developer происходит не потому, что модель так решила, а
потому что `verdict == "REVISION_NEEDED"` и счётчик меньше предела. Модель
решает ЧТО (вердикт), граф решает КУДА.

Отсюда и выбор LangGraph: здесь граф не декоративен. Есть развилка, есть
возврат назад, есть остановка на человеке — три вещи, ради которых он и нужен.

# Три инварианта, которые держит код, а не промпт

1. **Предел итераций.** `max_review_iterations` считается здесь. Промпт такое
   соблюдает обычно, но не всегда, а цена нарушения — бесконечный цикл на живых
   деньгах.
2. **Спецификацию утверждает человек.** До утверждения разработчик не получает
   ничего. Это защита не от плохой модели, а от дорогой ошибки: код по неверным
   требованиям стоит всех итераций сразу.
3. **QA не может починить.** У него нет `write_file` (см. `agents/roles.py`), и
   вернуть работу — единственное, что он может сделать с плохим кодом.

# Почему состояние плоское

`spec`, `code`, `review` лежат отдельными полями, а не одним `messages`. Так
видно, ЧТО именно передаётся между ролями, и так это можно проверить: тест
читает `state["review"].verdict`, а не разбирает последнее сообщение регулярками.
"""
from __future__ import annotations

import operator
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import observability
from agents.roles import BA, DEVELOPER, QA
from config import settings
from schemas import CodeOutput, ReviewOutput, SpecOutput


class TeamState(TypedDict, total=False):
    """Что команда знает о задаче в каждый момент."""

    user_story: str
    spec: SpecOutput | None
    code: CodeOutput | None
    review: ReviewOutput | None
    # Номер итерации ревью. Растёт только при возврате QA -> Developer.
    iteration: int
    # Замечания человека на спецификацию и замечания QA на код. Разные вещи, и
    # смешивать их нельзя: первое переписывает требования, второе — реализацию.
    spec_feedback: str
    review_feedback: str
    # История вердиктов: по ней видно, растёт ли качество между итерациями.
    scores: Annotated[list[float], operator.add]
    # Файл спецификации этой задачи. Заводится при первом проходе аналитика и
    # дальше только дополняется — см. `_save_spec`.
    spec_path: str
    # Номер версии спецификации: растёт с каждым возвратом от человека.
    spec_version: int


def _render_spec(spec: SpecOutput) -> str:
    """Спецификация текстом — и для человека у ворот, и для разработчика.

    Порядок не произвольный. Сначала ПЕРЕСКАЗ задачи и план работ: именно на них
    человек ловит нерасслышанное, и ловит с первых строк. Требования и критерии
    приёмки идут следом — они нужны, но проверять по ним понимание тяжело.

    Один и тот же вид у ворот и у разработчика тоже намеренно: человек
    утверждает ровно тот текст, который получит исполнитель, а не его пересказ.
    """
    lines = [f"TITLE: {spec.title}",
             f"COMPLEXITY: {spec.estimated_complexity}",
             "",
             "WHAT I UNDERSTOOD:",
             f"  {spec.restated_story}",
             "",
             "WORK PLAN:"]
    lines += [f"  {i}. {step}" for i, step in enumerate(spec.work_plan, 1)]
    lines += ["", "REQUIREMENTS:"]
    lines += [f"  {i}. {r}" for i, r in enumerate(spec.requirements, 1)]
    lines += ["", "ACCEPTANCE CRITERIA:"]
    lines += [f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)]
    return "\n".join(lines)


def _slug(text: str, limit: int = 40) -> str:
    """Имя файла из заголовка задачи. Латиница и цифры, остальное — дефис."""
    plain = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    plain = re.sub(r"[\s_]+", "-", plain)
    return (plain[:limit].strip("-") or "spec")


def _save_spec(state: TeamState, spec: SpecOutput) -> tuple[str, int]:
    """Записать спецификацию на диск. Новая задача — новый файл.

    # Почему дописываем, а не перезаписываем

    Каждый возврат от человека добавляет в тот же файл раздел «версия N». Так
    видно ИСТОРИЮ: что аналитик понял сначала, что человек попросил изменить и
    что получилось. Перезапись оставила бы только последнюю версию — и вопрос
    «почему в спеке появилось это требование» остался бы без ответа.

    Новая user story при этом получает новый файл: истории разных задач не
    смешиваются.

    # Почему пишет граф, а не аналитик

    У аналитика нет `write_file`, и это не случайность (см. `agents/roles.py`):
    аналитик, который пишет в проект, перестаёт быть аналитиком. Спецификацию
    сохраняет граф — как побочный эффект узла, а не как действие агента.
    """
    directory = Path(settings.specs_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = state.get("spec_path")
    version = state.get("spec_version", 0) + 1
    if not path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = str(directory / f"{stamp}-{_slug(spec.title)}.md")
        header = (f"# {spec.title}\n\n"
                  f"**User story:** {state['user_story']}\n")
        Path(path).write_text(header, encoding="utf-8")

    feedback = state.get("spec_feedback")
    section = [f"\n\n---\n\n## Версия {version}",
               f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"]
    if feedback:
        section.append(f"**Замечания человека к версии {version - 1}:** "
                       f"{feedback}\n")
    section.append(_render_spec(spec))
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(section) + "\n")
    return path, version


def _mark_approved(path: str | None, version: int) -> None:
    """Отметить в файле, какая именно версия ушла в разработку.

    Без этой отметки файл с тремя версиями не отвечает на главный вопрос: по
    какой из них писали код.
    """
    if not path:
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n---\n\n> ✅ **Версия {version} утверждена "
                     f"человеком {stamp}** и передана в разработку.\n")


def _render_review(review: ReviewOutput) -> str:
    lines = [f"VERDICT: {review.verdict}   score={review.score:.2f}"]
    if review.issues:
        lines.append("ISSUES:")
        lines += [f"  {i}. {x}" for i, x in enumerate(review.issues, 1)]
    if review.suggestions:
        lines.append("SUGGESTIONS:")
        lines += [f"  {i}. {x}" for i, x in enumerate(review.suggestions, 1)]
    return "\n".join(lines)


def _structured(result, model):
    """Достать модель данных из ответа LangGraph-агента.

    Проверка повторяется, хотя `response_format` уже обещал нужный тип: обещание
    не то же самое, что факт, и узнать о расхождении лучше здесь, чем в узле,
    который попытается прочитать несуществующее поле.
    """
    value = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(value, model):
        return value
    raise RuntimeError(
        f"агент вернул не {model.__name__}, а {type(value).__name__}. "
        f"Проверь response_format при сборке роли.")


def build(agents: dict, *, checkpointer=None):
    """Собрать граф. `agents` — уже готовые исполнители по именам ролей."""

    async def ba_node(state: TeamState) -> Command:
        """Аналитик пишет спецификацию. При повторном заходе — с замечаниями."""
        story = state["user_story"]
        feedback = state.get("spec_feedback")
        request = story if not feedback else (
            f"ORIGINAL USER STORY: {story}\n\n"
            f"The specification you produced was NOT approved. What the user "
            f"asks to change:\n{feedback}\n\n"
            f"Previous specification:\n{_render_spec(state['spec'])}\n\n"
            f"Rewrite the specification accordingly.")

        print("\n[BA] пишу спецификацию…", flush=True)
        result = await agents[BA.name].ainvoke(
            {"messages": [HumanMessage(content=request)]},
            config={"callbacks": observability.callbacks(), "run_name": "ba"})
        spec = _structured(result, SpecOutput)
        path, version = _save_spec(state, spec)
        print(f"  ← {len(spec.work_plan)} шагов плана, "
              f"{len(spec.requirements)} требований, "
              f"{len(spec.acceptance_criteria)} критериев приёмки, "
              f"сложность: {spec.estimated_complexity}", flush=True)
        print(f"  💾 {path}  (версия {version})", flush=True)
        return Command(goto="gate", update={
            "spec": spec, "spec_feedback": "",
            "spec_path": path, "spec_version": version})

    def gate_node(state: TeamState) -> Command:
        """Человек в контуре. Единственная остановка графа.

        `interrupt` не спрашивает человека сам — он ОСТАНАВЛИВАЕТ граф и отдаёт
        решение вызывающему. Поэтому здесь нет ни `input()`, ни печати вопроса:
        узел не знает, кто там снаружи — терминал, веб или тест.
        """
        spec: SpecOutput = state["spec"]
        answer = interrupt({
            "kind": "spec_approval",
            "title": spec.title,
            "spec": _render_spec(spec),
            "saved_to": state.get("spec_path"),
            "version": state.get("spec_version", 1),
        })

        if isinstance(answer, dict):
            decision = str(answer.get("decision", "")).lower()
            feedback = str(answer.get("feedback", ""))
        else:
            decision, feedback = str(answer).lower(), ""

        if decision.startswith("approve"):
            _mark_approved(state.get("spec_path"), state.get("spec_version", 1))
            return Command(goto="developer", update={"spec_feedback": ""})
        # Пустые замечания — это тоже сигнал: аналитик должен понять, что
        # именно не подошло, а «переделай» без указания чего бесполезно.
        return Command(goto="ba", update={
            "spec_feedback": feedback or "The user rejected the specification "
                                         "without saying why. Ask what is missing "
                                         "by listing open questions."})

    async def developer_node(state: TeamState) -> Command:
        """Разработчик пишет код. При возврате — с замечаниями ревью."""
        spec: SpecOutput = state["spec"]
        feedback = state.get("review_feedback")
        iteration = state.get("iteration", 0)

        if not feedback:
            request = (f"Implement this specification.\n\n{_render_spec(spec)}")
        else:
            request = (
                f"Your previous implementation was returned by review.\n\n"
                f"SPECIFICATION:\n{_render_spec(spec)}\n\n"
                f"REVIEW:\n{feedback}\n\n"
                f"Fix exactly what the review names. Keep what it did not "
                f"object to.")

        print(f"\n[Developer] пишу код…  (итерация {iteration + 1})", flush=True)
        result = await agents[DEVELOPER.name].ainvoke(
            {"messages": [HumanMessage(content=request)]},
            config={"callbacks": observability.callbacks(),
                    "run_name": "developer"})
        code = _structured(result, CodeOutput)
        print(f"  ← файлов: {', '.join(code.files_created) or '(ни одного)'}",
              flush=True)
        return Command(goto="qa", update={"code": code, "review_feedback": ""})

    async def qa_node(state: TeamState) -> Command:
        """QA проверяет код и решает судьбу итерации.

        Условное ребро живёт ЗДЕСЬ, а не в отдельной функции-роутере: решение
        принимается по вердикту и счётчику, и держать их в разных местах значит
        однажды поменять одно и забыть про другое.
        """
        spec: SpecOutput = state["spec"]
        code: CodeOutput = state["code"]
        iteration = state.get("iteration", 0)

        request = (f"Review this implementation against the specification.\n\n"
                   f"SPECIFICATION:\n{_render_spec(spec)}\n\n"
                   f"WHAT THE DEVELOPER SAYS:\n{code.description}\n\n"
                   f"FILES: {', '.join(code.files_created)}\n\n"
                   f"Read the files and run them before deciding.")

        print("\n[QA] проверяю…", flush=True)
        result = await agents[QA.name].ainvoke(
            {"messages": [HumanMessage(content=request)]},
            config={"callbacks": observability.callbacks(), "run_name": "qa"})
        review = _structured(result, ReviewOutput)
        print(f"  ← {review.verdict}  score={review.score:.2f}, "
              f"замечаний: {len(review.issues)}", flush=True)

        done = review.verdict == "APPROVED"
        exhausted = iteration + 1 >= settings.max_review_iterations
        update = {"review": review, "scores": [review.score]}

        if done or exhausted:
            if exhausted and not done:
                print(f"  ⏹  предел итераций ({settings.max_review_iterations}) "
                      f"исчерпан — отдаю как есть", flush=True)
            return Command(goto=END, update=update)

        return Command(goto="developer", update={
            **update,
            "iteration": iteration + 1,
            "review_feedback": _render_review(review),
        })

    graph = StateGraph(TeamState)
    graph.add_node("ba", ba_node)
    graph.add_node("gate", gate_node)
    graph.add_node("developer", developer_node)
    graph.add_node("qa", qa_node)
    graph.add_edge(START, "ba")
    # Остальные рёбра заданы через Command(goto=...) в самих узлах: маршрут
    # зависит от данных, и держать его рядом с решением честнее, чем разносить
    # по add_conditional_edges и отдельным функциям-роутерам.
    return graph.compile(checkpointer=checkpointer)
