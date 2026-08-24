#!/bin/sh
# Перенести из homework-lesson-10 то, что наблюдается БЕЗ изменений.
#
# hw12 не переписывает мультиагента — он делает его ВИДИМЫМ. Система здесь та же
# самая: MCP для инструментов, A2A для агентов, LangChain внутри агентов.
# Добавляется один слой — трассировка, промпты из Langfuse и online-оценка.
#
# Список явный, а не «скопировать каталог»: hw12 добавляет свой `observability.py`
# и свои скрипты, и любая автоматика с `--delete` однажды их затрёт.
#
# ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ:
#
# config.py — СВОЙ. В hw10 в нём лежат промпты; здесь их там быть НЕ ДОЛЖНО:
#   задание требует, чтобы ни один system prompt не был захардкожен в Python.
#   Промпты приезжают из Langfuse по имени и label.
#
# a2a_servers.py, a2a_utils.py, supervisor.py, main.py — СВОИ: в них живёт
#   проброс контекста трассировки между процессами и загрузка промптов.
#
# .env.example — свой: добавились ключи Langfuse и свои порты.
#
# data/ и index/ — свои копии, не симлинки. Работа должна быть целой сама по
#   себе, а корпус не должен меняться оттого, что переиндексировали соседа.
set -e
SRC="$(cd "$(dirname "$0")/../homework-lesson-10" && pwd)"
DST="$(cd "$(dirname "$0")" && pwd)"

FILES="embeddings.py importance.py ingest.py ocr.py preflight.py retriever.py
       schemas.py sparse.py tools.py translate.py vectorstore.py check_names.py
       requirements.txt"

OWN="config.py a2a_servers.py a2a_utils.py supervisor.py main.py .env.example
     observability.py prompts_seed.py start_servers.sh"

for f in $FILES; do
    if [ -e "$SRC/$f" ]; then cp -p "$SRC/$f" "$DST/$f"
    else echo "  ! нет в hw10: $f"; fi
done

for d in mcp_servers agents; do
    [ -d "$SRC/$d" ] && rsync -a --exclude '__pycache__' "$SRC/$d/" "$DST/$d/"
done

[ -f "$SRC/.private-names.txt" ] && cp -p "$SRC/.private-names.txt" "$DST/.private-names.txt"

echo "синхронизировано из hw10: $(echo $FILES | wc -w | tr -d ' ') файлов + mcp_servers/ agents/"
for f in $OWN; do
    if [ -e "$SRC/$f" ] && [ -e "$DST/$f" ] && ! diff -q "$SRC/$f" "$DST/$f" >/dev/null 2>&1; then
        echo "  свой (не тронут): $f — расходится с hw10, так и задумано"
    elif [ ! -e "$SRC/$f" ] && [ -e "$DST/$f" ]; then
        echo "  свой (не тронут): $f — у hw10 такого файла нет вовсе"
    fi
done
exit 0
