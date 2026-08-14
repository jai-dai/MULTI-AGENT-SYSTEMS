#!/bin/sh
# Отправить код из Agent_1 в репозиторий, откуда идёт публикация.
#
# Agent_1 — ОСНОВНОЕ место кодовой базы: здесь правится код, здесь же физически
# лежат индексы, почта и venv. Репозиторий `homework-lesson-5` держит git-историю
# и служит точкой публикации в my-mas, а тяжёлые каталоги там — симлинки сюда.
#
# Направление одностороннее. Правка, сделанная в репозитории, следующим запуском
# этого скрипта затрётся молча, и выглядеть это будет не как потеря, а как «оно
# почему-то перестало работать».
#
#   ./sync_to_repo.sh          — показать, что поедет
#   ./sync_to_repo.sh --go     — отправить
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="/Users/dhanika/MULTI-AGENT-SYSTEMS/homework-lesson-5"

# .env у каждого места свой: в репозитории он всё равно не публикуется, а вот
# затереть им рабочий было бы неприятно. Индексы, почта и venv не копируются —
# они там симлинки СЮДА, и rsync с --delete снёс бы их вместе с данными.
EXCLUDE="--exclude .venv --exclude .env --exclude index --exclude index_vl
         --exclude index_mail --exclude mail --exclude output --exclude __pycache__
         --exclude *.log --exclude .git"

if [ "$1" = "--go" ]; then
    rsync -a --delete $EXCLUDE "$SRC/" "$DST/"
    echo "отправлено в $DST"
    echo "дальше: git -C /Users/dhanika/MULTI-AGENT-SYSTEMS add homework-lesson-5 && git commit"
else
    rsync -an --delete --itemize-changes $EXCLUDE "$SRC/" "$DST/" \
        | grep -v '^\.d\.\.t' || echo "различий нет"
    echo
    echo "(пробный прогон — повторить с --go)"
fi
