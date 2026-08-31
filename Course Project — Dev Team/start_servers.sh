#!/bin/sh
# Поднять всё, что нужно команде, и не поднимать дважды.
#
# Процессов четыре: Qdrant, DocsMCP, WorkspaceMCP и REPL с графом. Порты
# читаются из .env — сервер и скрипт обязаны брать адрес из одного места,
# иначе скрипт ждёт не там, не дожидается и бодро печатает «готово».
#
#   ./start_servers.sh          — поднять
#   ./start_servers.sh --stop   — погасить свои серверы (Qdrant общий, не трогаем)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
QDRANT_START="/Users/dhanika/Documents/VL/qdrant/start.sh"

port_of() {
    value=$(grep -E "^$1=" "$HERE/.env" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' \r')
    echo "${value:-$2}"
}
DOCS=$(port_of DOCS_MCP_PORT 8931)
WORKSPACE=$(port_of WORKSPACE_MCP_PORT 8932)

alive() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if [ "$1" = "--stop" ]; then
    for port in "$DOCS" "$WORKSPACE"; do
        pid=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        [ -n "$pid" ] && kill $pid && echo "  погашен порт $port"
    done
    exit 0
fi

if alive 6333; then
    echo "  6333 Qdrant        — уже поднят"
else
    nohup "$QDRANT_START" >/dev/null 2>&1 &
    printf "  6333 Qdrant        — поднимаю"
    for _ in $(seq 1 30); do alive 6333 && break; printf .; sleep 1; done
    echo " готово"
fi

start() {
    port="$1"; script="$2"; name="$3"
    if alive "$port"; then echo "  $port $name — уже поднят"; return 0; fi
    nohup "$PY" -u "$HERE/$script" > "$HERE/${name}.log" 2>&1 &
    printf "  %s %s — поднимаю" "$port" "$name"
    ok=0
    for _ in $(seq 1 90); do alive "$port" && { ok=1; break; }; printf .; sleep 1; done
    [ "$ok" = 1 ] && echo " готово" || echo " НЕ ПОДНЯЛСЯ — смотри ${name}.log"
}

start "$DOCS" mcp_servers/docs_mcp.py DocsMCP
start "$WORKSPACE" mcp_servers/workspace_mcp.py WorkspaceMCP

echo
echo "запустить:  .venv/bin/python main.py"
echo "погасить:   ./start_servers.sh --stop"
