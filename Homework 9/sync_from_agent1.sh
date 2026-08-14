#!/bin/sh
# Перенести из Agent_1 то, что переиспользуется БЕЗ изменений.
#
# Agent_1 — основное место кодовой базы агента. Правки делаются ТАМ и
# приезжают сюда; обратное направление запрещено.
#
# Список явный, а не «скопировать всё»: hw8 добавляет своих агентов и свой
# супервизор, и любая автоматика вида `rsync каталог целиком` однажды затрёт их.
# Расхождение в retriever.py выглядит не как ошибка, а как «поиск стал хуже»,
# поэтому копии не должны разъезжаться молча.
#
# Индексы и data/ не копируются, а слинкованы: пересборка стоит часов, а вторая
# копия — сотен мегабайт.
set -e
SRC="/Users/dhanika/Documents/VL/Agent_1"
DST="$(dirname "$0")"

FILES="llm.py retriever.py vectorstore.py embeddings.py sparse.py translate.py
       ocr.py importance.py ingest.py preflight.py config.py tools.py
       check_names.py .gitignore requirements.txt"

for f in $FILES; do
    if [ -e "$SRC/$f" ]; then
        cp -p "$SRC/$f" "$DST/$f"
    else
        echo "  ! нет в hw5: $f"
    fi
done

for d in mailprep hooks; do
    [ -d "$SRC/$d" ] && rsync -a --delete --exclude '__pycache__' "$SRC/$d/" "$DST/$d/"
done

# Список приватных имён не копируется в git, но нужен локально стражу.
[ -f "$SRC/.private-names.txt" ] && cp -p "$SRC/.private-names.txt" "$DST/.private-names.txt"

echo "синхронизировано из Agent_1: $(echo $FILES | wc -w | tr -d ' ') файлов + mailprep/ + hooks/"
