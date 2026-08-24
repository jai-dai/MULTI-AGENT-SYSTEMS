"""Важность письма — по правилам из weights.yaml, а не по статистике.

Почему не статистика, подробно написано в самом `weights.yaml`; коротко: по
частоте переписки первыми идут рассылки, а взаимность отделяет живого
корреспондента от шума, но ставит типографию выше нотариуса. Важность — это
суждение о бизнесе, и в переписке оно не записано.

Что здесь есть, чего нет в конфиге: **объяснение**. `score()` возвращает не
только число, но и список причин, из которых оно сложилось. Цифра важности без
объяснения непроверяема — её нельзя ни оспорить, ни поправить, и она молча
переживёт любое изменение в делах.

    score = вес контрагента × множитель роли × множитель темы
            + бонус за взаимность + бонус за вложение

Множители умножают, бонусы прибавляют, и это не стилистика. Множитель роли
поднимает то, что и так весомо: письмо от директора типографии про визитки не
становится важным оттого, что он директор. Бонус за взаимность прибавляется,
чтобы переписка с типографией не обгоняла налоговую, которой я никогда не пишу.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

WEIGHTS_FILE = "weights.yaml"
# Вес контрагента задаётся в шкале 0..100, но ИТОГОВЫЙ балл не обрезается по
# ней. Обрезался в первой версии — и двенадцать писем в топе получили ровно 100
# каждое: базовые веса и так 85-95, множители выбивали за границу, и
# ранжирование вырождалось именно там, где оно нужнее всего. Шкала здесь
# относительная, и 190 честнее прижатой сотни.
WEIGHT_SCALE = 100


def _stem_match(haystack: str, patterns: list[str]) -> bool:
    """Совпадение по НАЧАЛУ слова, а не по любому месту внутри него.

    Простая подстрока подводит в обе стороны. `test` находился внутри `latest`
    — заміряно, тема «releases latest Clarity Act text» попадала в «рутину» с
    множителем 0.6. А сузить до точного слова тоже нельзя: `заборгован` обязан
    ловить «заборгованості», иначе украинские падежи придётся перечислять
    руками, и первый же непредусмотренный отправит письмо со сроком вниз.

    Граница слова слева плюс свободный хвост справа даёт и то, и другое: основа
    работает, `latest` больше не считается тестом.

    Цена честная: составные слова так не находятся — `оплат` не совпадёт с
    «передоплата». Такие формы вписываются в конфиг отдельной строкой, и это
    лучше случайных попаданий.
    """
    return any(re.search(r"\b" + re.escape(str(pattern).strip()), haystack,
                         re.IGNORECASE | re.UNICODE)
               for pattern in patterns if str(pattern).strip())


def _match(value: str, patterns: list[str]) -> bool:
    """Совпадение по адресу целиком или по домену, с шаблонами fnmatch."""
    value = (value or "").lower()
    if not value:
        return False
    domain = value.split("@")[-1]
    for raw in patterns:
        pattern = str(raw).lower()
        if fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(domain, pattern):
            return True
        # Домен без звёздочек трактуется как «этот домен и его поддомены»:
        # писать `*.a**n.ua` в конфиге каждый раз — лишний повод ошибиться.
        if "*" not in pattern and (domain == pattern or domain.endswith("." + pattern)):
            return True
    return False


def load(directory: str | Path | None = None) -> dict:
    """Правила из weights.yaml. Пустой словарь, если файла нет.

    Отсутствие файла — не ошибка: без него важность просто не считается, и
    `list_mail` продолжает отдавать письма по дате. Молча падать из-за
    ненастроенной необязательной функции хуже, чем её не иметь.

    А вот файл, который ЕСТЬ и не разбирается, — ошибка, и громкая. Разница
    принципиальная: «весов нет» и «веса настроены, но не применились» выглядят
    в выдаче одинаково — всё ровно по дате, — но второе означает, что человек
    расставил важность, а система её потеряла. Одна лишняя двоеточие в `why`
    обнуляла бы весь файл до `default_weight`, сообщив об этом одной строкой в
    потоке лога.
    """
    path = Path(directory or Path(__file__).parent) / WEIGHTS_FILE
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(
            f"{path} не разбирается: {type(exc).__name__}: {exc}\n"
            f"  Веса заданы, но применить их нельзя. Работать дальше значило бы "
            f"тихо выдать всем один и тот же вес по умолчанию.\n"
            f"  Частая причина — двоеточие с пробелом внутри незакавыченного "
            f"текста и значение, начинающееся со звёздочки (в YAML это алиас)."
        ) from exc


def is_noise(address: str, rules: dict) -> bool:
    return _match(address, rules.get("noise") or [])


def counterparty(address: str, rules: dict) -> tuple[str, float]:
    """(имя контрагента, вес). Неизвестный адрес получает default_weight."""
    for entry in rules.get("counterparties") or []:
        if _match(address, entry.get("match") or []):
            return entry.get("name", address), float(entry.get("weight", 0))
    return "", float(rules.get("default_weight", 20))


def weight_of(address: str, rules: dict) -> float:
    """Вес 0..1 — для мягкого буста в поиске."""
    if not rules or is_noise(address, rules):
        return 0.0
    return counterparty(address, rules)[1] / WEIGHT_SCALE


def score(message: dict, rules: dict, wrote_to: set[str] | None = None) -> dict:
    """Важность письма и объяснение, из чего она сложилась.

    `message` — словарь как из `mailprep.store.list_messages`.
    `wrote_to` — адреса, которым писал сам пользователь: считается один раз по
    всей базе, а не заново для каждого письма.
    """
    if not rules:
        return {"score": 0, "reasons": [], "counterparty": "", "noise": False}

    sender = (message.get("sender_email") or "").lower()
    haystack = " ".join([
        message.get("subject") or "",
        " ".join(message.get("attachments") or []),
        message.get("body_preview") or "",
    ]).lower()

    # Тема считается ДО списка шума. Порядок обратный интуиции и в этом весь
    # смысл: уведомление госоргана о задолженности приходит автоматической
    # рассылкой, и фильтр, поставленный против «Ведомостей», выбросил бы
    # единственное письмо, пропуск которого стоит денег.
    topic = None
    for entry in rules.get("topics") or []:
        if _stem_match(haystack, entry.get("match") or []):
            topic = entry
            break                       # одна тема на письмо — самая первая

    name, base = counterparty(sender, rules)

    # Пробить список шума темой может только ИЗВЕСТНЫЙ контрагент. Первая версия
    # разрешала это любому отправителю, и рассылка The Block немедленно въехала
    # в топ с баллом 85, употребив слово из списка сроков. Разрешение было шире
    # намерения: смысл в том, что уведомление госоргана приходит автоматической
    # рассылкой, а не в том, что любое упоминание срока делает письмо важным.
    if is_noise(sender, rules) and not (
            name and (topic or {}).get("overrides_noise")):
        return {"score": 0, "reasons": ["в списке шума"], "counterparty": "",
                "noise": True}

    reasons = [f"{name or 'неизвестный отправитель'}: {base:.0f}"]

    if topic:
        multiplier = float(topic.get("multiplier", 1))
        base *= multiplier
        reasons.append(f"тема «{topic.get('name', '?')}»: ×{multiplier}")

    signature = (message.get("body_preview") or "")[-400:].lower()
    for entry in rules.get("roles") or []:
        if _stem_match(signature, entry.get("match") or []):
            multiplier = float(entry.get("multiplier", 1))
            base *= multiplier
            reasons.append(f"должность в подписи: ×{multiplier}")
            break

    tuning = rules.get("tuning") or {}
    if wrote_to and sender in wrote_to:
        bonus = float(tuning.get("reciprocity_bonus", 0))
        base += bonus
        reasons.append(f"я сам ему писал: +{bonus:.0f}")
    if message.get("attachments"):
        bonus = float(tuning.get("attachment_bonus", 0))
        base += bonus
        reasons.append(f"есть вложения: +{bonus:.0f}")

    # Порог снизу применяется ПОСЛЕДНИМ, поверх всех множителей и бонусов: он
    # выражает «сколько бы ни насчиталось, ниже этого такое письмо не опускать»,
    # а не участвует в арифметике наравне с остальными.
    floor = float((topic or {}).get("floor", 0))
    if floor and base < floor:
        reasons.append(f"порог темы «{topic.get('name')}»: не ниже {floor:.0f}")
        base = floor

    return {"score": round(base), "reasons": reasons,
            "counterparty": name, "noise": False}
