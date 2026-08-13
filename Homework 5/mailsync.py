"""Спросить при старте, забирать ли новую почту, и забрать — до запуска агента.

# Почему именно при старте, а не инструментом агента

У агента нет инструмента, который ходит в почту, и это не недоделка. Причин
три, и все три снимаются тем, что докачка происходит ЗДЕСЬ:

  * Встроенный Qdrant держит каталог одним процессом. Инструмент, который
    дописывает индекс во время работы агента, конфликтовал бы с ним же —
    именно так `knowledge_search` и падал с "already accessed by another
    instance". Здесь запись заканчивается раньше, чем агент откроет индексы.
  * Пароль от почты остаётся у `main.py`. Агент читает письма, а письма пишут
    посторонние люди: инструкция, подброшенная в текст письма, не должна иметь
    доступа к ящику.
  * Решение «грузить или нет» принимает человек, а не модель по ходу
    рассуждения. Стоимость шага человеку видна заранее — число писем названо.

# Почему вопрос дешёвый

Проба — это `UID SEARCH` без выкачки тел, секунды. Вопрос, который сам стоит
минуту ожидания, человек начнёт пропускать не глядя, и тогда весь замысел
теряется.
"""
from __future__ import annotations

import sys

from config import settings

# Выше этого — уже не «пара писем между делом», и человеку стоит сказать, во
# что это обойдётся, ПРЕЖДЕ чем он согласится. Порог низкий намеренно: цена
# согласия должна быть видна до нажатия, а не после.
BULK_HINT = 30

# Всё замерено на этой машине и на этом корпусе 2026-08-12. Согласие стоит
# ТРЁХ вещей, а не одной, и оценка по одной только выкачке врала бы втрое.
#
#   выкачка       5286 писем за 1962 с. Тела идут целиком, вложения по дороге
#                 пишутся на диск — отсюда и цена.
#   конвейер      5 с на всю базу из 1183 писем. Фиксированная добавка: чанки
#                 режутся заново каждый раз, даже когда новых писем два. Растёт
#                 с размером базы, а не с числом новых.
#   эмбеддинг     1129 чанков за 3917 с — 0.29 чанка/с. Это ВТРОЕ медленнее,
#                 чем те же 0.79 на вложениях, и это не ошибка замера: чанк
#                 письма несёт шапку и целое тело, документный режется по 500
#                 символов. Брать скорость от одного корпуса для другого —
#                 ровно тот способ занизить оценку втрое.
FETCH_PER_SECOND = 2.7
PIPELINE_SECONDS = 5.0
CHUNKS_PER_MESSAGE = 0.95
CHUNKS_PER_SECOND = 0.29

# Столько за раз — это уже не накопившаяся почта, а первая выкачка или смена
# UIDVALIDITY. Такое делается осознанно отдельной командой.
BACKFILL_HINT = 500


def _cost(messages: int) -> str:
    """Во что обойдётся согласие — словами, а не в чанках.

    Считается ВЕСЬ путь до готового поиска: забрать письма, перерезать базу на
    чанки, посчитать векторы новым. Оценка по одной выкачке была бы честной
    только для того, кто закроет терминал сразу после неё.
    """
    seconds = (messages / FETCH_PER_SECOND
               + PIPELINE_SECONDS
               + messages * CHUNKS_PER_MESSAGE / CHUNKS_PER_SECOND)
    if seconds < 90:
        return f"около {max(round(seconds), 5)} с"
    return f"около {round(seconds / 60)} мин"


def _ask(question: str) -> bool:
    """Да/нет с настройкой на «нет».

    Умолчание отрицательное намеренно: Enter, нажатый не глядя, не должен
    запускать сетевую работу и запись в индекс.
    """
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes", "д", "да")


def offer_refresh() -> None:
    """Проверить почту и, с согласия человека, добрать её в индекс.

    Ничего не роняет: любая беда по дороге — это сообщение и продолжение работы
    с тем индексом, который уже есть. Агент, который не стартует из-за почты,
    хуже агента с чуть устаревшей почтой.
    """
    if not settings.mail_index_dir:
        return
    if not settings.imap_user or not settings.imap_password:
        return
    # Неинтерактивный запуск (пайп, cron, тесты) — спрашивать некого, а
    # молча лезть в сеть за человека тем более не стоит.
    if not sys.stdin.isatty():
        return

    from mailprep.imap_fetch import count_new

    print("проверяю почту…", end=" ", flush=True)
    found = count_new(settings.imap_host, settings.imap_user,
                      settings.imap_password.get_secret_value(),
                      settings.imap_folder_list, settings.mail_db)
    if found["error"]:
        print(f"не вышло ({found['error']}) — работаю с тем, что в индексе")
        return
    total = found["total"]
    if not total:
        print("новых писем нет")
        return

    where = ", ".join(f"{name}: {n}" for name, n in found["folders"].items())
    print(f"новых писем: {total} ({where})")
    if total > BACKFILL_HINT:
        print(f"   ! {total} за раз — это не накопившаяся почта, а первая "
              "выкачка. Её лучше делать отдельно и осознанно: "
              "python -m mailprep.imap_fetch --backfill --since ГГГГ-ММ-ДД")
    elif total > BULK_HINT:
        print(f"   ! больше {BULK_HINT} писем — агент не стартует, пока "
              "не досчитает")
    # Цена в самом вопросе, а не только в предупреждении: она нужна и на пяти
    # письмах — согласие в любом случае означает ожидание, и знать сколько
    # полезно до нажатия, а не после.
    if not _ask(f"Загрузить {total} ({_cost(total)}) и добавить в поиск?"):
        print("пропускаю — работаю с тем, что в индексе")
        return

    try:
        _refresh()
    except SystemExit as exc:                # осознанный отказ изнутри конвейера
        print(f"   ! докачка остановлена: {exc}")
    except Exception as exc:
        print(f"   ! докачка не удалась ({type(exc).__name__}: {exc}) — "
              "работаю с тем, что в индексе")


def _refresh() -> None:
    """Забрать письма и дописать только новые чанки.

    Вложения сюда НЕ входят: они идут в индекс документов через `ingest.py`,
    там своя инкрементальность по хешам файлов, и время на OCR несопоставимо —
    это отдельное решение, а не часть «проверить почту».
    """
    from mailprep import index as mail_index
    from mailprep.imap_fetch import sync

    sync(settings.imap_host, settings.imap_user,
         settings.imap_password.get_secret_value(),
         settings.imap_folder_list, settings.mail_db,
         settings.imap_since or None,
         attachments_dir=settings.mail_attachments_dir,
         max_mb=settings.mail_attachment_max_mb)

    result = mail_index.build(only_new=True, index_name=settings.mail_index_dir)
    if result["embedded"]:
        print(f"добавлено в поиск: {result['embedded']} чанков "
              f"(всего {result['chunks']})")
