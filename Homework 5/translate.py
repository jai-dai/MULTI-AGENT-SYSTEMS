"""Переклад запиту під мову пасажа — щоб реранкер узагалі мав що оцінювати.

Проблема не в пошуку, а в реранкері, і вона заміряна. Питання «чому
постачальник затримав відповідь по котлу» проти англомовного листування дає
`bge-reranker-base` **весь пул на 0.0000-0.0001**: він не зіставляє мови між
собою. Той самий зміст англійською — 0.8976 і 0.8604 на потрібних пасажах.
Реранкер не помилявся в оцінці, він її не мав.

Ембеддер (`bge-m3`) мультимовний і цю пару тягне сам — потрібний лист він
знаходить і ставить другим. Тому переклад іде **тільки в реранкінг**: пошук
залишається на вихідному запиті. Перекладений запит у BM25 втратив би
збіги за іменами, номерами договорів і кодами — саме тим, заради чого лексична
гілка й існує.

Альтернативою була мультимовна модель `bge-reranker-v2-m3` — 2.2 ГБ проти
нинішніх 1.1 на машині з 8 ГБ, де поряд лежить індекс. Переклад коштує
десятки токенів на запит і нуль памʼяті.
"""
from __future__ import annotations

import re

# Письменності, які реально трапляються в цьому корпусі. Перелік не
# теоретичний: заміряно на обох індексах — 4094 чанки латиницею, 3362
# кирилицею і 14 корейською (довіреності контрагента, лист до посольства).
# Китайська трапляється лише впереміш із латиницею в листуванні з
# постачальниками і переважною не буває, але клас лишається — зʼявиться
# китайський документ, і його не доведеться відносити «до латиниці».
#
# Повноцінне визначення мови (langdetect, fasttext) додало б залежність заради
# різниці, якої тут немає: реранкеру байдуже українська перед ним чи російська,
# він падає саме на переході між абетками.
_SCRIPTS = {
    "cyrillic": re.compile(r"[а-яіїєґёА-ЯІЇЄҐЁ]"),
    "latin": re.compile(r"[a-zA-Z]"),
    "hangul": re.compile(r"[가-힯ᄀ-ᇿ]"),
    "han": re.compile(r"[一-鿿]"),
    "kana": re.compile(r"[぀-ヿ]"),
}

# Людські назви — для повідомлення користувачеві, а не для моделі.
SCRIPT_NAMES = {
    "cyrillic": "українською або російською",
    "latin": "латиницею",
    "hangul": "корейською",
    "han": "китайською",
    "kana": "японською",
}

# Мова, якою просимо переклад для кожної письменності.
_TARGET_NAME = {
    "latin": "English",
    "cyrillic": "Ukrainian",
    "hangul": "Korean",
    "han": "Chinese",
    "kana": "Japanese",
}

# Що реранкер узагалі здатен оцінити. `bge-reranker-base` навчений на
# англійській і китайській; кирилицю він тягне помітно гірше, але тягне, а
# корейську та японську — ні. Для них переклад запиту не рятує, і чесніше
# сказати про це у видачі, ніж мовчки віддати оцінку, якої не існує.
RERANKER_SCRIPTS = {"latin", "han", "cyrillic"}

_cache: dict[tuple[str, str], str] = {}


def script_of(text: str) -> str:
    """Переважна письменність тексту: ключ із `_SCRIPTS`.

    Порожній або цифровий текст рахується латиницею — не тому, що це правда, а
    тому, що для такого тексту переклад не має сенсу, і латиниця тут означає
    «нічого не робити».
    """
    counts = {name: len(pattern.findall(text)) for name, pattern in _SCRIPTS.items()}
    top = max(counts, key=counts.get)
    return top if counts[top] else "latin"


def unsupported(script: str) -> str | None:
    """Повідомлення про мову, яку реранкер оцінити не може, або None.

    Це не помилка й не привід ховати пасаж: документ знайдено, він у видачі, і
    користувач має знати, що впорядкований він гірше за решту — і що читати
    його доведеться з окремим перекладом.
    """
    if script in RERANKER_SCRIPTS:
        return None
    return (f"документ {SCRIPT_NAMES.get(script, script)} — реранкер цю мову не "
            "оцінює, порядок для нього орієнтовний")


_PROMPT = (
    "Translate the search query below into {target}. Output ONLY the "
    "translation — no quotes, no explanation, no alternatives.\n\n"
    "Keep proper names, company names, product codes and numbers exactly as "
    "written in the original: they are the same in both languages, and "
    "translating them would destroy the match.\n\nQuery: {query}"
)


def translate(query: str, target: str) -> str | None:
    """Запит цільовою абеткою, або None якщо переклад не вдався.

    Помилка перекладу НЕ ламає пошук: викликач лишається з вихідним запитом і
    працює як раніше. Додатковий крок, який здатен покласти пошук, коштував би
    більше за проблему, яку розвʼязує.
    """
    key = (query, target)
    if key in _cache:
        return _cache[key]

    try:
        from llm import get_backend
        reply = get_backend().complete(
            [{"role": "user",
              "content": _PROMPT.format(target=_TARGET_NAME[target], query=query)}],
            None)
        text = (reply.text or "").strip()
    except Exception as exc:                 # мережа, ключ, ліміт — не наша біда
        print(f"   ! переклад запиту не вдався: {type(exc).__name__}: {exc}")
        text = ""

    result = text if text and script_of(text) == target else None
    _cache[key] = result
    return result
