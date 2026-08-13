#!/bin/sh
# How far has a running `ingest.py` got, when its own output is not visible?
#
# The embedding phase is the slow one (hours), and with EMBEDDING_BACKEND=compat
# every batch is one POST to Ollama — which Ollama logs with a timestamp. So the
# batches already answered are countable from outside the process, without
# touching it. Multiply by EMBED_BATCH_SIZE and that is chunks embedded.
#
# usage:  ./progress.sh [HH:MM:SS when ingest started]   (default: today 00:00)
LOG=~/.ollama/logs/server.log
BATCH=${EMBED_BATCH_SIZE:-64}
START=${1:-00:00:00}

awk -F'|' '/POST     "\/v1\/embeddings"/ {split($1, a, " - "); split(a[2], t, " "); print t[1]}' "$LOG" \
  | awk -v s="$START" '$1 >= s' > /tmp/ingest_batches.txt

N=$(wc -l < /tmp/ingest_batches.txt | tr -d ' ')
[ "$N" -eq 0 ] && { echo "no embedding requests since $START"; exit 0; }

python3 - "$N" "$BATCH" "$(head -1 /tmp/ingest_batches.txt)" "$(tail -1 /tmp/ingest_batches.txt)" <<'PY'
import sys
from datetime import datetime
n, batch, first, last = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
mins = (datetime.strptime(last, "%H:%M:%S") - datetime.strptime(first, "%H:%M:%S")).total_seconds() / 60
rate = n * batch / (mins * 60) if mins else 0
print(f"embedding since {first}, last batch at {last} ({mins:.0f} min)")
print(f"{n} batches x {batch} = ~{n * batch} chunks embedded, {rate:.2f} chunks/s")
print("the total is only known to the process itself — it prints `embedded X/Y`")
PY
