"""BM25 як розріджений вектор — щоб лексичний пошук жив у сховищі, а не в RAM.

Спокуслива причина — «`rank_bm25` будується при кожному старті» — заміром НЕ
підтвердилась: 0.39 секунди на 7347 чанків. Справжніх причин три, і жодна з них
не про швидкість старту:

1. **Фільтр діє ДО скорингу.** У розрідженого вектора в Qdrant той самий
   payload-фільтр, що й у щільного, тож пошук по одному файлу чи по листуванню
   з одним адресатом віддає потрібним чанкам увесь top-K. `rank_bm25` про
   payload не знає нічого: він ранжує весь корпус, а зайве відкидається після.
2. **49 МБ RAM** — заміряно, стільки важить токенізована копія корпусу в
   памʼяті процесу. На машині з 8 ГБ, де поряд лежить реранкер на 1.1 ГБ, це
   не дрібниця.
3. Обидві ціни ростуть лінійно з корпусом, а тут вони зникають зовсім.

Qdrant уміє розріджені вектори нативно, і BM25 кладеться в них без втрат.
Формула розкладається на дві половини:

    score(q, d) = Σ  idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1−b+b·|d|/avgdl))
                 t∈q

Права частина залежить ТІЛЬКИ від документа, тому обчислюється один раз при
індексації і лягає вагою терміна у вектор документа. Запит тоді — вектор з
одиниць, а скалярний добуток дає той самий BM25. Термін, якого немає у
словнику, не дає внеску — це правильна поведінка, а не втрата.

Токенізатор тут один на весь проєкт (`retriever` імпортує його звідси)
свідомо: два токенізатори, що розійшлися, вже одного разу вимкнули стадію BM25
цілком і зробили це мовчки.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

VOCAB_FILE = "sparse_vocab.json"

# `\w` з re.UNICODE, а не `[a-z0-9]`: ASCII-клас лишає від «Статут ТОВ АТОН-ГРУП
# нова редакція 2020» один токен `2020`. Історія в README.
_TOKEN = re.compile(r"\w+", re.UNICODE)

K1 = 1.5     # насичення частотою — канонічне значення BM25
B = 0.75     # наскільки довжина документа штрафує вагу


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def build(chunks: list[dict]) -> tuple[dict[str, list], list[dict[int, float]]]:
    """(словник, розріджені вектори документів) для всього корпусу.

    Словник: токен -> [індекс у векторі, idf]. Індекс довільний, але мусить
    пережити перезапуск процесу — інакше запит адресуватиме інші виміри, ніж ті,
    в які писалися документи, і пошук поверне впевнене сміття.
    """
    tokenised = [tokenize(chunk["text"]) for chunk in chunks]
    total = len(tokenised) or 1
    average_length = sum(len(t) for t in tokenised) / total

    frequency = Counter()
    for tokens in tokenised:
        frequency.update(set(tokens))

    vocabulary = {
        token: [position,
                # +1 усередині логарифма тримає idf додатним навіть для терміна,
                # що є в кожному документі: інакше найчастіші слова отримали б
                # відʼємну вагу і почали віднімати від score.
                math.log(1 + (total - count + 0.5) / (count + 0.5))]
        for position, (token, count) in enumerate(frequency.items())
    }

    vectors = []
    for tokens in tokenised:
        counts = Counter(tokens)
        length = len(tokens)
        vector = {}
        for token, tf in counts.items():
            position, idf = vocabulary[token]
            norm = tf + K1 * (1 - B + B * length / (average_length or 1))
            vector[position] = idf * tf * (K1 + 1) / (norm or 1)
        vectors.append(vector)
    return vocabulary, vectors


def query_vector(query: str, vocabulary: dict[str, list]) -> dict[int, float]:
    """Запит -> розріджений вектор, вага = скільки разів термін ужито в запиті.

    Не одиниця за унікальний токен, і це не дрібниця. Заміряно: з одиницями
    набір давав 13/17 hit@1 проти 14/17 у `rank_bm25`, і розійшлися вони саме
    там, де термін у запиті повторювався — «advisory agreement Sokrat Financial
    **Advisory**». `rank_bm25` рахує такий термін двічі, бо підсумовує по
    токенах запиту, а не по множині. Уся вага документа вже в його векторі,
    тому тут лишається тільки частота в запиті — і тоді формули збігаються.
    """
    counts = Counter(token for token in tokenize(query) if token in vocabulary)
    return {vocabulary[token][0]: float(count) for token, count in counts.items()}


def save(directory: Path, vocabulary: dict[str, list]) -> None:
    (Path(directory) / VOCAB_FILE).write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")


def load(directory: Path) -> dict[str, list] | None:
    path = Path(directory) / VOCAB_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
