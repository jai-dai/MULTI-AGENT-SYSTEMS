#!/bin/sh
# Записать полный baseline: все стадии по всему golden dataset.
#
# Порядок не произвольный — он совпадает с зависимостями стадий (см. DEPENDS в
# tests/capture.py) и идёт от дешёвого к дорогому. Если прогон оборвётся,
# повторный запуск продолжит с места обрыва: уже записанные стадии пропускаются.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
export DEEPEVAL_TELEMETRY_OPT_OUT=1

# Два процесса capture одновременно писали бы один и тот же runs/<id>.json
# чтением-изменением-записью целиком, и один затёр бы другого.
while pgrep -f "tests.capture" > /dev/null; do
    echo "$(date '+%H:%M:%S')  жду, пока освободится capture..."
    sleep 20
done

stage() {
    echo ""
    echo "=========================================================="
    echo "$(date '+%H:%M:%S')  СТАДИЯ: $*"
    echo "=========================================================="
    "$PY" -m tests.capture "$@" || echo "  (стадия завершилась с ошибками — продолжаю)"
}

stage --stage planner   --all
stage --stage researcher --category happy_path
stage --stage critic     --category happy_path
# e2e последним и по всему датасету: самая дорогая стадия, и к моменту её
# запуска все дешёвые записи уже на диске — обрыв здесь не потеряет ничего.
stage --stage e2e        --all

echo ""
echo "$(date '+%H:%M:%S')  baseline записан"
