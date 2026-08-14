#!/bin/sh
# Поднять всё, что нужно hw9, и не поднимать дважды.
#
# Процессов теперь четыре, и это не усложнение ради протокола: MCP-серверы
# держат модели (кросс-энкодер 1.1 ГБ живёт в SearchMCP один раз на всех), а
# Qdrant общий для всех индексов. Каждый проверяется по порту, поэтому повторный
# запуск безопасен — скрипт поднимет только то, чего нет.
#
#   ./start_servers.sh          — поднять
#   ./start_servers.sh --stop   — погасить MCP-серверы (Qdrant не трогаем)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
QDRANT_START="/Users/dhanika/Documents/VL/qdrant/start.sh"

alive() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if [ "$1" = "--stop" ]; then
    for port in 8901 8902 8903; do
        pid=$(lsof -nP -tiTCP:$port -sTCP:LISTEN 2>/dev/null || true)
        [ -n "$pid" ] && kill $pid && echo "  погашен порт $port"
    done
    exit 0
fi

# Qdrant — общий и для агента, и для мультиагента, поэтому не гасится вместе с
# ними и запускается отдельным скриптом.
if alive 6333; then
    echo "  6333 Qdrant       — уже поднят"
else
    nohup "$QDRANT_START" >/dev/null 2>&1 &
    printf "  6333 Qdrant       — поднимаю"
    for _ in $(seq 1 30); do alive 6333 && break; printf .; sleep 1; done
    echo " готово"
fi

# Порядок здесь ни на что не влияет: ACP-сервер подключается к SearchMCP лениво,
# при первом обращении к агенту, а не при старте. Это осознанно — сервер,
# падающий оттого, что сосед ещё не поднялся, невозможно запускать в
# произвольном порядке.
for entry in "8901:mcp_servers/search_mcp.py:SearchMCP" \
             "8902:mcp_servers/report_mcp.py:ReportMCP" \
             "8903:acp_server.py:ACP"; do
    port=${entry%%:*}; rest=${entry#*:}; script=${rest%%:*}; name=${rest#*:}
    if alive "$port"; then
        echo "  $port $name    — уже поднят"
        continue
    fi
    # Логи в файл: серверы говорчивые, и их вывод не должен лезть в диалог.
    nohup "$PY" -u "$HERE/$script" > "$HERE/${name}.log" 2>&1 &
    printf "  %s %s    — поднимаю" "$port" "$name"
    for _ in $(seq 1 40); do alive "$port" && break; printf .; sleep 1; done
    echo " готово"
done

echo
echo "проверить:  curl -s localhost:6333/healthz && echo ok"
echo "погасить:   ./start_servers.sh --stop"
