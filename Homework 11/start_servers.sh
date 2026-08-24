#!/bin/sh
# Поднять всё, что нужно hw10, и не поднимать дважды.
#
# Отличие от hw9 одно, и оно не косметическое: ПОРТЫ ЧИТАЮТСЯ ИЗ .env, а не
# зашиты здесь. У hw10 они свои (8911/8912/8913), чтобы обе работы поднимались
# одновременно и не мешали друг другу.
#
# Почему это вообще понадобилось. Пока порты совпадали, серверы протоколов
# поднимались из кода hw9, а e2e-прогон hw10 уходил на них — то есть тесты
# измеряли СОСЕДНЮЮ систему. Поймано на исправлении, которое работало на
# компонентном уровне и никак не проявлялось в сквозном: планировщик hw10 умел
# отказываться, планировщик на порту 8903 — нет, потому что это был чужой
# планировщик.
#
# Зашитый список портов при этом ещё и ВРАЛ: серверы читали .env и вставали на
# 8911+, а скрипт ждал их на 8901+, сорок секунд не дожидался и печатал
# «готово». Отсюда правило: адрес сервера должен браться из одного места с тем,
# что читает сам сервер.
#
#   ./start_servers.sh          — поднять
#   ./start_servers.sh --stop   — погасить свои серверы (Qdrant не трогаем)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
QDRANT_START="/Users/dhanika/Documents/VL/qdrant/start.sh"

# Из .env, с теми же значениями по умолчанию, что у config.py.
port_of() {
    value=$(grep -E "^$1=" "$HERE/.env" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' \r')
    echo "${value:-$2}"
}
SEARCH_PORT=$(port_of SEARCH_MCP_PORT 8901)
REPORT_PORT=$(port_of REPORT_MCP_PORT 8902)
ACP_PORT_=$(port_of ACP_PORT 8903)

alive() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if [ "$1" = "--stop" ]; then
    for port in "$SEARCH_PORT" "$REPORT_PORT" "$ACP_PORT_"; do
        pid=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        [ -n "$pid" ] && kill $pid && echo "  погашен порт $port"
    done
    exit 0
fi

# Qdrant общий для всех работ и не гасится вместе с ними.
if alive 6333; then
    echo "  6333 Qdrant       — уже поднят"
else
    nohup "$QDRANT_START" >/dev/null 2>&1 &
    printf "  6333 Qdrant       — поднимаю"
    for _ in $(seq 1 30); do alive 6333 && break; printf .; sleep 1; done
    echo " готово"
fi

# Порядок не важен: ACP-сервер подключается к SearchMCP лениво, при первом
# обращении к агенту, а не на старте.
for entry in "$SEARCH_PORT:mcp_servers/search_mcp.py:SearchMCP" \
             "$REPORT_PORT:mcp_servers/report_mcp.py:ReportMCP" \
             "$ACP_PORT_:acp_server.py:ACP"; do
    port=${entry%%:*}; rest=${entry#*:}; script=${rest%%:*}; name=${rest#*:}
    if alive "$port"; then
        echo "  $port $name    — уже поднят"
        continue
    fi
    nohup "$PY" -u "$HERE/$script" > "$HERE/${name}.log" 2>&1 &
    printf "  %s %s    — поднимаю" "$port" "$name"
    ok=0
    for _ in $(seq 1 60); do alive "$port" && { ok=1; break; }; printf .; sleep 1; done
    # Молчаливое «готово» на не поднявшемся сервере — это ложь, которая потом
    # выглядит как «система сломалась». Говорим прямо.
    [ "$ok" = 1 ] && echo " готово" || echo " НЕ ПОДНЯЛСЯ — смотри ${name}.log"
done

echo
echo "проверить:  curl -s localhost:6333/healthz && echo ok"
echo "погасить:   ./start_servers.sh --stop"
