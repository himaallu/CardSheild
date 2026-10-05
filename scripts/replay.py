"""Stream test-period rows to the scoring API (normal or drifted traffic).

python scripts/replay.py                # 2,000 rows at 100 req/s
python scripts/replay.py --drift        # second half: Amount x3, V14/V17 +2 std
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence

import httpx

from cardshield.config import INPUT_COLS, RAW_CSV
from cardshield.data import load_raw, time_split
from cardshield.monitoring.drift import WINDOW, apply_drift, drift_shifts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", "--rows", type=int, default=WINDOW)
    parser.add_argument("--rate", type=float, default=100.0, help="requests per second")
    parser.add_argument("--start", type=int, default=0, help="first test-period row")
    parser.add_argument("--drift", action="store_true", help="simulate drift in second half")
    args = parser.parse_args(argv)

    split = time_split(load_raw(RAW_CSV))
    rows = split.test.iloc[args.start : args.start + args.rows].reset_index(drop=True)
    if args.drift:
        rows = apply_drift(rows, drift_shifts(split.train))
    payloads = [{c: float(r[c]) for c in INPUT_COLS} for _, r in rows.iterrows()]

    mode = "DRIFTED (2nd half)" if args.drift else "normal"
    print(f"Replaying {len(payloads)} {mode} test rows to {args.url} at {args.rate:g} req/s")
    decisions: Counter[str] = Counter()
    interval, next_at = 1.0 / args.rate, time.perf_counter()
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for i, payload in enumerate(payloads, 1):
            next_at += interval
            response = client.post("/v1/score", json=payload)
            response.raise_for_status()
            decisions[response.json()["decision"]] += 1
            if i % 500 == 0 or i == len(payloads):
                print(f"  {i}/{len(payloads)} sent  {dict(decisions)}")
            time.sleep(max(0.0, next_at - time.perf_counter()))
    print(f"Done: {decisions['block']} blocked, {decisions['approve']} approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
