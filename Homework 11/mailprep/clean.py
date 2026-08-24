"""
Очистка тела письма: цитаты, подписи, дисклеймеры, служебный мусор.

ЭТО САМЫЙ ВАЖНЫЙ ФАЙЛ ПРОЕКТА. Качество RAG по почте определяется здесь,
а не в выборе эмбеддинг-модели.

Подход: набор эвристик, каждая — отдельная функция, применяются по порядку.
Все паттерны вынесены в константы наверху — их надо ДОПОЛНЯТЬ под свои данные.
Смотри dump-выгрузку своих 200-300 писем и добавляй то, что реально встречается.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# --------------------------------------------------------------------------
# ПАТТЕРНЫ — расширять под свою переписку
# --------------------------------------------------------------------------

# Заголовки цитируемого письма ("On ... wrote:", "От: ...", корейские варианты).
# Всё, что НИЖЕ такой строки, обычно является цитатой.
QUOTE_HEADERS = [
    # английские
    r"^\s*On .{5,120}\s+wrote:\s*$",
    r"^\s*On .{5,120},\s*$",                      # перенос строки в длинном "On ..."
    r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$",
    r"^\s*_{5,}\s*$",                              # разделитель Outlook
    r"^\s*From:\s*.+$",
    r"^\s*Sent:\s*.+$",
    # русские
    r"^\s*\d{1,2}\s+\S+\s+\d{4}\s*г?\.?,?\s*.{0,80}(написал|писал)\(?а?\)?:?\s*$",
    r"^\s*От:\s*.+$",
    r"^\s*Отправлено:\s*.+$",
    r"^\s*-{2,}\s*Исходное сообщение\s*-{2,}\s*$",
    r"^\s*Кому:\s*.+$",
    # украинские
    r"^\s*Від:\s*.+$",
    r"^\s*Надіслано:\s*.+$",
    r"^\s*Кому:\s*.+$",
    r"^\s*\d{1,2}\s+\S+\s+\d{4}\s*р?\.?,?\s*.{0,80}(написав|писав)\(?ла?\)?:?\s*$",
    # корейские (для переписки с Doosan / GS / Woojin)
    r"^\s*보낸\s*사람:\s*.+$",
    r"^\s*받는\s*사람:\s*.+$",
    r"^\s*보낸\s*날짜:\s*.+$",
]

# Разделители подписи. Всё ниже — подпись.
SIGNATURE_MARKERS = [
    r"^--\s*$",                    # RFC 3676 стандартный разделитель
    r"^\s*--\s*$",
    r"^\s*—\s*$",
    r"^\s*Best regards,?\s*$",
    r"^\s*Kind regards,?\s*$",
    r"^\s*Yours (faithfully|sincerely),?\s*$",
    r"^\s*(Sincerely|Regards|Thanks|Thank you),?\s*$",
    r"^\s*С уважением,?\s*$",
    r"^\s*З повагою,?\s*$",
    r"^\s*Всего доброго,?\s*$",
    r"^\s*З найкращими побажаннями,?\s*$",
    r"^\s*감사합니다\.?\s*$",
]

# Мобильные футеры — удаляем всегда
MOBILE_FOOTERS = [
    r"^\s*Sent from my (iPhone|iPad|Android|Samsung|Galaxy).*$",
    r"^\s*Get Outlook for (iOS|Android).*$",
    r"^\s*Отправлено из мо(бильного|его) .*$",
    r"^\s*Надіслано з .*$",
]

# Корпоративные дисклеймеры. Если строка матчится — режем ОТ НЕЁ И ДО КОНЦА.
DISCLAIMER_STARTS = [
    r"This (e-?mail|message) (and any attachments? )?(is|are) (confidential|intended)",
    r"CONFIDENTIALITY NOTICE",
    r"DISCLAIMER\s*:",
    r"The information contained in this (e-?mail|transmission)",
    r"If you (are not|have received) th(is|e) (e-?mail|message) in error",
    r"P(lease)?\.?\s*consider the environment before printing",
    r"Данное сообщение (и любые вложения )?(является|содержит) конфиденциальн",
    r"Это письмо и любые приложения к нему",
    r"Це повідомлення .{0,40}конфіденц",
    r"본\s*메일은",                       # корейский дисклеймер
]

# Строки, которые режем всегда, где бы ни встретились
NOISE_LINES = [
    r"^\s*\[cid:[^\]]+\]\s*$",              # inline-картинки
    r"^\s*<https?://\S+>\s*$",              # голая ссылка в угловых скобках
    r"^\s*\[image:[^\]]*\]\s*$",
    r"^\s*ВНИМАНИЕ:\s*Это письмо пришло",   # антифишинговые баннеры
    r"^\s*CAUTION:\s*This email originated",
    r"^\s*EXTERNAL( EMAIL)?\s*:",
]

# Строка целиком из цитаты (начинается с > )
QUOTED_LINE = re.compile(r"^\s*>+\s?")

# --------------------------------------------------------------------------

_QUOTE_HEADER_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in QUOTE_HEADERS]
_SIGNATURE_RE = [re.compile(p, re.IGNORECASE) for p in SIGNATURE_MARKERS]
_MOBILE_RE = [re.compile(p, re.IGNORECASE) for p in MOBILE_FOOTERS]
_DISCLAIMER_RE = [re.compile(p, re.IGNORECASE) for p in DISCLAIMER_STARTS]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_LINES]


class CleanResult(NamedTuple):
    body: str
    quoted_removed_chars: int
    signature_removed: bool
    disclaimer_removed: bool


def strip_quoted_lines(lines: list[str]) -> tuple[list[str], int]:
    """
    Убирает строки, начинающиеся с '>'.
    Возвращает (оставшиеся строки, сколько символов срезано).
    """
    kept, removed = [], 0
    for ln in lines:
        if QUOTED_LINE.match(ln):
            removed += len(ln)
        else:
            kept.append(ln)
    return kept, removed


def cut_at_quote_header(lines: list[str]) -> tuple[list[str], int]:
    """
    Ищет первый заголовок цитируемого письма ("On ... wrote:", "От: ...")
    и обрезает всё от него до конца.

    ВАЖНО: ищем только начиная со 2-й строки — иначе пересланное письмо,
    начинающееся с "From:", схлопнется в пустоту.
    """
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        for rx in _QUOTE_HEADER_RE:
            if rx.match(ln):
                removed = sum(len(x) for x in lines[i:])
                return lines[:i], removed
    return lines, 0


def cut_signature(lines: list[str]) -> tuple[list[str], bool]:
    """
    Обрезает подпись. Ищем маркер С КОНЦА — подпись всегда внизу,
    а слово 'Regards' может встретиться в середине текста.

    Эвристика: маркер должен быть в последних 15 строках.
    """
    window_start = max(0, len(lines) - 15)
    for i in range(len(lines) - 1, window_start - 1, -1):
        for rx in _SIGNATURE_RE:
            if rx.match(lines[i]):
                return lines[:i], True
    return lines, False


def cut_disclaimer(lines: list[str]) -> tuple[list[str], bool]:
    """Режет корпоративный дисклеймер от места старта и до конца письма."""
    for i, ln in enumerate(lines):
        for rx in _DISCLAIMER_RE:
            if rx.search(ln):
                return lines[:i], True
    return lines, False


def drop_noise(lines: list[str]) -> list[str]:
    """Удаляет служебные строки где угодно в теле."""
    out = []
    for ln in lines:
        if any(rx.match(ln) for rx in _NOISE_RE):
            continue
        if any(rx.match(ln) for rx in _MOBILE_RE):
            continue
        out.append(ln)
    return out


def collapse_blank_lines(text: str) -> str:
    """3+ пустых строки -> 1. Убирает хвостовые пробелы."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_body(raw: str) -> CleanResult:
    """
    Главная функция. Порядок операций важен:

    1. noise    — убрать баннеры/cid до того, как они собьют другие эвристики
    2. quote header — обрезать хвост с цитируемым письмом
    3. quoted lines — добить оставшиеся '>' строки
    4. disclaimer   — режем юридический хвост
    5. signature    — режем подпись (последней, т.к. она ближе всего к концу)
    """
    if not raw:
        return CleanResult("", 0, False, False)

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    lines = drop_noise(lines)

    lines, removed_a = cut_at_quote_header(lines)
    lines, removed_b = strip_quoted_lines(lines)
    lines, disclaimer = cut_disclaimer(lines)
    lines, signature = cut_signature(lines)

    body = collapse_blank_lines("\n".join(lines))
    return CleanResult(body, removed_a + removed_b, signature, disclaimer)


# --------------------------------------------------------------------------
# Нормализация темы и HTML
# --------------------------------------------------------------------------

_SUBJ_PREFIX = re.compile(
    r"^\s*((re|fwd?|fw|aw|sv|вх|отв|пересыл|перес|відп|пересил|답장|전달)\s*(\[\d+\])?\s*:\s*)+",
    re.IGNORECASE,
)


def normalize_subject(subject: str) -> str:
    """'Re: Fwd: RE: Панели' -> 'Панели'. Для группировки цепочек."""
    if not subject:
        return ""
    prev = None
    cur = subject
    while prev != cur:
        prev = cur
        cur = _SUBJ_PREFIX.sub("", cur).strip()
    return cur


def html_to_text(html: str) -> str:
    """
    Fallback, если у письма нет text/plain.
    Наивно, но для писем обычно достаточно. Если качество не устроит —
    заменить на BeautifulSoup(html, 'lxml').get_text('\\n').
    """
    if not html:
        return ""
    html = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|tr|li|h[1-6])>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", "\t", html)
    text = re.sub(r"(?s)<[^>]+>", "", html)
    # html-сущности
    for a, b in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                 ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(a, b)
    return collapse_blank_lines(text)


# --------------------------------------------------------------------------
# Определение языка (грубое, по алфавиту — без внешних зависимостей)
# --------------------------------------------------------------------------

_UK_ONLY = set("іїєґІЇЄҐ")
_RU_ONLY = set("ыъэёЫЪЭЁ")
_CYRILLIC = re.compile(r"[а-яА-ЯёЁіїєґІЇЄҐ]")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_lang(text: str) -> str:
    """
    Возвращает ru / uk / en / ko / mixed / "".
    Нужно для отладки и для фильтров в Qdrant. bge-m3 сам мультиязычный,
    так что на качество эмбеддинга не влияет.
    """
    if not text.strip():
        return ""
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    kor = len(_HANGUL.findall(text))
    total = cyr + lat + kor
    if total == 0:
        return ""

    scores = {"cyr": cyr / total, "lat": lat / total, "ko": kor / total}
    top = max(scores, key=scores.get)

    if scores[top] < 0.6:
        return "mixed"
    if top == "ko":
        return "ko"
    if top == "lat":
        return "en"
    # кириллица: различаем uk / ru по характерным буквам
    chars = set(text)
    if chars & _UK_ONLY:
        return "uk"
    if chars & _RU_ONLY:
        return "ru"
    return "ru"
