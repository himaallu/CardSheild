"""Async latency benchmark: p50/p95/p99 and requests/second against a running API.

Single requests at concurrency 1 are the S7 number (p95 < 50 ms); concurrency 8 shows
queueing on one server process; batches of 100 show throughput per transaction.
"""

import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from cardshield.config import INPUT_COLS, RAW_CSV


@dataclass(frozen=True)
class BenchResult:
    name: str
    requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    rps: float

    def row(self) -> str:
        return (
            f"{self.name:<28}{self.requests:>7}{self.p50_ms:>9.1f}{self.p95_ms:>9.1f}"
            f"{self.p99_ms:>9.1f}{self.rps:>10.1f}"
        )


HEADER = f"{'scenario':<28}{'reqs':>7}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'req/s':>10}"


def summarize(name: str, latencies_ms: Sequence[float], wall_s: float) -> BenchResult:
    lat = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    return BenchResult(name, len(lat), float(p50), float(p95), float(p99), len(lat) / wall_s)


def sample_transactions(n: int) -> list[dict[str, float]]:
    """Real rows if the dataset is present, otherwise standard-normal synthetic rows."""
    if RAW_CSV.exists():
        import pandas as pd

        frame = pd.read_csv(RAW_CSV, usecols=list(INPUT_COLS)).sample(n, random_state=0)
        return [{c: float(r[c]) for c in INPUT_COLS} for _, r in frame.iterrows()]
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(n, len(INPUT_COLS)))
    return [
        {
            c: float(abs(v)) if c in ("Time", "Amount") else float(v)
            for c, v in zip(INPUT_COLS, r, strict=True)
        }
        for r in rows
    ]


async def run_scenario(
    client: httpx.AsyncClient, name: str, path: str, payloads: Sequence[Any], concurrency: int
) -> BenchResult:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for p in payloads:
        queue.put_nowait(p)
    latencies: list[float] = []

    async def worker() -> None:
        while not queue.empty():
            payload = queue.get_nowait()
            start = time.perf_counter()
            response = await client.post(path, json=payload)
            latencies.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return summarize(name, latencies, time.perf_counter() - start)


async def bench(url: str, n: int, batch_size: int, n_batches: int) -> list[BenchResult]:
    txs = sample_transactions(max(n, batch_size))
    batches = [{"transactions": txs[:batch_size]}] * n_batches
    async with httpx.AsyncClient(base_url=url, timeout=30.0) as client:
        await run_scenario(client, "warmup", "/v1/score", txs[:50], 1)
        return [
            await run_scenario(client, "single, concurrency 1", "/v1/score", txs[:n], 1),
            await run_scenario(client, "single, concurrency 8", "/v1/score", txs[:n], 8),
            await run_scenario(
                client, f"batch of {batch_size}, conc. 1", "/v1/score/batch", batches, 1
            ),
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", "--requests", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batches", type=int, default=30)
    args = parser.parse_args(argv)
    results = asyncio.run(bench(args.url, args.requests, args.batch_size, args.batches))
    print(HEADER)
    for r in results:
        print(r.row())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
