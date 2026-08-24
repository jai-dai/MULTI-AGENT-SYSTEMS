#!/bin/sh
# Перенести из homework-lesson-9 то, что тестируется БЕЗ изменений.
#
# hw10 не переписывает мультиагента — он его измеряет. Поэтому система здесь
# должна быть той же самой, и правится она ТАМ, в hw9; обратное направление
# запрещено.
#
# Цепочка копий: Agent_1 → hw9 → hw10, но звенья РАЗНЫЕ по смыслу.
# Agent_1 → hw9 — это общий слой поиска между двумя разными проектами, и оттуда
# приезжает только он (config.py, например, у hw9 уже свой). hw9 → hw10 — это
# система под тестом, и она должна совпадать ЦЕЛИКОМ, включая config.py:
# тест, гоняющий не ту конфигурацию, которая работает в бою, проверяет соседа.
#
# Список явный, а не «скопировать каталог»: hw10 добавляет свои tests/, runs/ и
# свой README, и любая автоматика вида `rsync целиком --delete` однажды их
# затрёт.
#
# --------------------------------------------------------------------------
# ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ
# --------------------------------------------------------------------------
#
# requirements-eval.txt — единственная зависимость, которой у hw9 нет и быть не
#   должно (deepeval откатывает click и rich, см. README). Приезжай он из hw9,
#   его бы там не было, и он исчезал бы при каждой синхронизации — молча.
#
# .env.example — у hw10 своя переменная JUDGE_MODEL_NAME: судья обязан быть
#   другим вендором, чем система, и это настройка hw10, а не hw9.
#
# data/ и index/ — БОЛЬШЕ НЕ СИМЛИНКИ (2026-08-24). Свои копии: домашка должна
#   быть целой сама по себе, а корпус под тестами — не меняться оттого, что
#   кто-то переиндексировал соседний проект. Для регрессионных тестов это не
#   удобство, а условие: индекс, изменившийся между прогонами, делает
#   зафиксированные в runs/ прогоны несопоставимыми.
#
#   ВНИМАНИЕ: векторы живут в СЕРВЕРЕ Qdrant и разделяются ИМЕНЕМ КОЛЛЕКЦИИ, а
#   не каталогом. Пока в .env стоит `INDEX_DIR=index`, hw8/hw9/hw10 работают с
#   одной коллекцией `index`. Каталоги теперь свои, коллекция — ещё нет.
set -e
SRC="$(cd "$(dirname "$0")/../homework-lesson-9" && pwd)"
DST="$(cd "$(dirname "$0")" && pwd)"

FILES="acp_compat.py acp_server.py acp_utils.py bridge.py check_names.py
       config.py embeddings.py importance.py ingest.py llm.py main.py
       mcp_utils.py ocr.py preflight.py prompts.py retriever.py schemas.py
       sparse.py structured.py supervisor.py tools.py translate.py
       vectorstore.py start_servers.sh
       requirements.txt requirements-protocols.txt"

# Намеренно НЕ синхронизируются. Скрипт про них не молчит.
OWN=".env.example requirements-eval.txt"

missing=0
for f in $FILES; do
    if [ -e "$SRC/$f" ]; then
        cp -p "$SRC/$f" "$DST/$f"
    else
        echo "  ! нет в hw9: $f"
        missing=$((missing + 1))
    fi
done
chmod +x "$DST/start_servers.sh"

for d in agents mcp_servers mailprep hooks; do
    [ -d "$SRC/$d" ] && rsync -a --delete --exclude '__pycache__' "$SRC/$d/" "$DST/$d/"
done

[ -f "$SRC/.private-names.txt" ] && cp -p "$SRC/.private-names.txt" "$DST/.private-names.txt"

echo "синхронизировано из hw9: $(echo $FILES | wc -w | tr -d ' ') файлов + agents/ mcp_servers/ mailprep/ hooks/"
[ "$missing" -gt 0 ] && echo "  ВНИМАНИЕ: не найдено файлов: $missing"

for f in $OWN; do
    if [ -e "$SRC/$f" ] && [ -e "$DST/$f" ] && ! diff -q "$SRC/$f" "$DST/$f" > /dev/null 2>&1; then
        added=$(diff "$SRC/$f" "$DST/$f" | grep -c '^>' || true)
        echo "  свой (не тронут): $f — здесь на $added строк больше, чем в hw9"
    elif [ ! -e "$SRC/$f" ]; then
        echo "  свой (не тронут): $f — у hw9 такого файла нет вовсе"
    fi
done
exit 0
