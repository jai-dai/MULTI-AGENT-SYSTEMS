"""QA: инварианты вердикта кодом, полезность ревью — судьёй.

Инварианты здесь не декоративны. `REVISION_NEEDED` без единой претензии — это
тупик: разработчик получает «переделай» без указания чего, и итерация сгорает
впустую. `APPROVED` с длинным списком проблем — обратная беда: работа уходит
дальше, а замечания теряются.

Оба случая ДЕТЕРМИНИРОВАНЫ, и проверяет их `assert` — бесплатно и однозначно.
Судье достаётся то, что требует суждения: конкретна ли претензия и можно ли по
ней что-то сделать.
"""
from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from tests import metrics
from tests.conftest import ids_of, run_or_skip, stories
from tests.scored import assert_scored
from tests.context import render_files, render_requirements

SOLID = stories("happy_path", "medium")


def _render(review: dict) -> str:
    lines = [f"VERDICT: {review['verdict']}   score={review['score']}"]
    if review["issues"]:
        lines.append("ISSUES:\n" + "\n".join(f"  - {i}" for i in review["issues"]))
    if review["suggestions"]:
        lines.append("SUGGESTIONS:\n"
                     + "\n".join(f"  - {s}" for s in review["suggestions"]))
    return "\n".join(lines)


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_verdict_is_actionable(story):
    """Вердикт согласован с содержимым ревью."""
    review = run_or_skip(story["id"])["review"]
    assert review is not None, "QA не вернул вердикт"
    if review["verdict"] == "REVISION_NEEDED":
        assert review["issues"], (
            "REVISION_NEEDED без единой претензии — разработчик получит "
            "«переделай» без указания чего, и итерация сгорит впустую")
        assert review["suggestions"], "нет ни одного предложения, что делать"
    else:
        assert not review["issues"], (
            "APPROVED, но список проблем не пуст — работа уйдёт дальше, "
            "а замечания потеряются")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_score_matches_verdict(story):
    """Оценка и вердикт не должны противоречить друг другу.

    `score` не участвует в маршрутизации, поэтому разойтись с вердиктом может
    незаметно. APPROVED при 0.3 означает, что одно из двух посчитано формально.
    """
    review = run_or_skip(story["id"])["review"]
    if review["verdict"] == "APPROVED":
        assert review["score"] >= 0.6, (
            f"APPROVED при score={review['score']} — вердикт не следует "
            f"из собственной оценки")


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_review_is_useful(story, judge):
    record = run_or_skip(story["id"])
    spec = record["spec"]
    context = (render_requirements(spec["requirements"])
               + "\n\nCODE:\n"
               + render_files(record["files"]))
    assert_scored(
        LLMTestCase(input=context, actual_output=_render(record["review"])),
        [metrics.review_is_useful(judge)])


@pytest.mark.parametrize("story", SOLID, ids=ids_of(SOLID))
def test_review_is_complete(story, judge):
    """QA не должен подписывать код, который нарушает требование."""
    record = run_or_skip(story["id"])
    context = (render_requirements(record["spec"]["requirements"])
               + "\n\nCODE:\n"
               + render_files(record["files"]))
    assert_scored(
        LLMTestCase(input=context, actual_output=_render(record["review"])),
        [metrics.review_completeness(judge)])
