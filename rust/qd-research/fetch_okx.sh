#!/usr/bin/env bash
# Fetch real OHLCV candles from the OKX public API (no key needed) and write
# them in the strict CSV format qd-research consumes:
#   t,open,high,low,close,volume
#
# Usage:
#   fetch_okx.sh INST_ID BAR LIMIT OUTFILE
# Example:
#   fetch_okx.sh BTC-USDT 1D 1000 /tmp/btc_1d.csv
#
# BAR uses OKX granularity tokens: 1m 3m 5m 15m 30m 1H 2H 4H 6H 12H 1D 1W.
# Pagination walks backwards via `after=` until LIMIT ascending bars are
# collected. Any HTTP/parse failure aborts non-zero — never write partial
# data and call it history.
set -euo pipefail

INST="${1:?usage: fetch_okx.sh INST_ID BAR LIMIT OUTFILE}"
BAR="${2:?usage: fetch_okx.sh INST_ID BAR LIMIT OUTFILE}"
LIMIT="${3:?usage: fetch_okx.sh INST_ID BAR LIMIT OUTFILE}"
OUT="${4:?usage: fetch_okx.sh INST_ID BAR LIMIT OUTFILE}"

TMP_RAW="$(mktemp)"
trap 'rm -f "$TMP_RAW" "$TMP_RAW.page"' EXIT

AFTER=""
GOT=0
: > "$TMP_RAW"
while [ "$GOT" -lt "$LIMIT" ]; do
  NEED=$((LIMIT - GOT))
  PAGE=100
  [ "$NEED" -lt "$PAGE" ] && PAGE="$NEED"
  URL="https://www.okx.com/api/v5/market/history-candles?instId=${INST}&bar=${BAR}&limit=${PAGE}"
  [ -n "$AFTER" ] && URL="${URL}&after=${AFTER}"
  if ! curl -s -m 20 "$URL" -o "$TMP_RAW.page"; then
    echo "fetch_okx: curl failed for $URL" >&2
    exit 1
  fi
  python3 - "$TMP_RAW.page" "$TMP_RAW" <<'EOF'
import json, sys
page_path, acc_path = sys.argv[1], sys.argv[2]
with open(page_path) as f:
    payload = json.load(f)
if str(payload.get("code")) != "0":
    sys.stderr.write("fetch_okx: API error: %s\n" % json.dumps(payload)[:300])
    sys.exit(1)
rows = payload.get("data") or []
with open(acc_path, "a") as f:
    for c in rows:
        # [ts_ms, o, h, l, c, vol, ...] newest-first; t in whole seconds.
        f.write("%d,%s,%s,%s,%s,%s\n" % (int(int(c[0]) // 1000), c[1], c[2], c[3], c[4], c[5]))
print(len(rows))
EOF
  NROWS=$(python3 - "$TMP_RAW.page" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
print(len(payload.get("data") or []))
EOF
)
  if [ "${NROWS:-0}" -eq 0 ]; then
    echo "fetch_okx: no more history (got $GOT/$LIMIT)" >&2
    break
  fi
  GOT=$((GOT + NROWS))
  AFTER=$(python3 - "$TMP_RAW.page" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
rows = payload.get("data") or []
print(rows[-1][0])
EOF
)
done

# Ascending by t, de-duplicated, truncated to the most recent LIMIT.
python3 - "$TMP_RAW" "$OUT" "$LIMIT" <<'EOF'
import sys
acc_path, out_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = {}
with open(acc_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        t = int(line.split(",", 1)[0])
        seen[t] = line  # latest fetch wins on duplicates
bars = [seen[t] for t in sorted(seen)]
bars = bars[-limit:]
if len(bars) < 2:
    sys.stderr.write("fetch_okx: too few bars (%d), refusing to write\n" % len(bars))
    sys.exit(1)
# Strictly increasing guard before anything touches the pipeline.
prev = None
for b in bars:
    t = int(b.split(",", 1)[0])
    if prev is not None and t <= prev:
        sys.stderr.write("fetch_okx: non-increasing t=%d after %d\n" % (t, prev))
        sys.exit(1)
    prev = t
with open(out_path, "w") as f:
    f.write("t,open,high,low,close,volume\n")
    for b in bars:
        f.write(b + "\n")
print("wrote %d bars to %s" % (len(bars), out_path))
EOF
