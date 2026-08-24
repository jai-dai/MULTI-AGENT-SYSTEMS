#!/bin/sh
# Поднять всё, что нужно hw10, и не поднимать дважды.
#
# Процессов теперь ШЕСТЬ: Qdrant, два MCP-сервера и три A2A-агента (последние
# живут в одном процессе, но на трёх портах — снаружи это три независимых
# агента с тремя карточками).
#
# Порты читаются из .env, а не зашиты здесь: сервер и скрипт обязаны брать адрес
# из одного места, иначе скрипт ждёт не там, не дожидается и врёт «готово».
#
#   ./start_servers.sh          — поднять
#   ./start_servers.sh --stop   — погасить (Qdrant не трогаем, он общий)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
QDRANT_START="/Users/dhanika/Documents/VL/qdrant/start.sh"

port_of() {
    value=$(grep -E "^$1=" "$HERE/.env" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' \r')
    echo "${value:-$2}"
}
SEARCH=$(port_of SEARCH_MCP_PORT 8921)
REPORT=$(port_of REPORT_MCP_PORT 8922)
PLANNER=$(port_of PLANNER_PORT 8923)
RESEARCHER=$(port_of RESEARCHER_PORT 8924)
CRITIC=$(port_of CRITIC_PORT 8925)

alive() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if [ "$1" = "--stop" ]; then
    for port in "$SEARCH" "$REPORT" "$PLANNER" "$RESEARCHER" "$CRITIC"; do
        pid=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        [ -n "$pid" ] && kill $pid && echo "  погашен порт $port"
    done
    exit 0
fi

if alive 6333; then
    echo "  6333 Qdrant       — уже поднят"
else
    nohup "$QDRANT_START" >/dev/null 2>&1 &
    printf "  6333 Qdrant       — поднимаю"
    for _ in $(seq 1 30); do alive 6333 && break; printf .; sleep 1; done
    echo " готово"
fi

start() {   # порт, скрипт, имя, порт-для-проверки
    port="$1"; script="$2"; name="$3"; check="${4:-$1}"
    if alive "$check"; then echo "  $port $name — уже поднят"; return 0; fi
    nohup "$PY" -u "$HERE/$script" > "$HERE/${name}.log" 2>&1 &
    printf "  %s %s — поднимаю" "$port" "$name"
    ok=0
    for _ in $(seq 1 90); do alive "$check" && { ok=1; break; }; printf .; sleep 1; done
    [ "$ok" = 1 ] && echo " готово" || echo " НЕ ПОДНЯЛСЯ — смотри ${name}.log"
}

start "$SEARCH" mcp_servers/search_mcp.py SearchMCP
start "$REPORT" mcp_servers/report_mcp.py ReportMCP
# Три агента поднимаются одним процессом, но проверять надо ПОСЛЕДНИЙ порт:
# если жив critic, значит поднялись все три.
start "$PLANNER,$RESEARCHER,$CRITIC" a2a_servers.py A2A "$CRITIC"

echo
echo "карточки:  curl -s localhost:$PLANNER/.well-known/agent-card.json"
echo "погасить:  ./start_servers.sh --stop"
