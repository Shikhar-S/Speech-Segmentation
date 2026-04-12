"""Shared helper: write a records list as sharded parquet with embedded audio.

Used by thchs30/librispeech/cv data-convert scripts to produce output in the
HF Hub ``{out}/data/{split}-{idx:05d}-of-{total:05d}.parquet`` layout. That
layout is what ``datasets.load_dataset(<dir>)`` auto-detects, so the resulting
directory is directly loadable by the existing ``SegmentationDataModule``.

We bypass ``Dataset.save_to_disk`` for two reasons:

1. save_to_disk writes ``*.arrow`` shards in its own directory layout, which
   is NOT loadable by ``load_dataset`` — you need ``load_from_disk``. Our
   dataloader uses ``load_dataset``, so we need parquet output.
2. save_to_disk's embed-external-files step is single-process by default; for
   our scale (hundreds of thousands of audio files on NFS), embedding bytes
   per-row serially is ~hours. Sharding over ``multiprocessing.Pool`` cuts
   that to minutes.

Workers receive only ``(start, end, out_path)`` tuples and read their slice
from a module-level records list inherited from the parent via fork's COW
memory. This avoids pickling millions of dicts through the multiprocessing
input queue, which deadlocked the cv_ali run (4.7M records / 2368 shards).
"""

from __future__ import annotations

import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pyarrow.parquet as pq
from datasets.table import embed_table_storage

log = logging.getLogger(__name__)

# Pin numerical-backend threads inside workers to 1 so 64 parallel workers
# don't oversubscribe cores. Must be set before any numerical import.
_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Module-level handles inherited by forked workers (no pickling cost).
_RECORDS: List[Dict] = []
_FEATURES: datasets.Features = None  # type: ignore[assignment]


def _init_worker() -> None:
    for var in _THREAD_LIMIT_VARS:
        os.environ.setdefault(var, "1")


def _write_shard(args: Tuple[int, int, str]) -> int:
    """Worker: slice global records, embed audio bytes, write parquet shard."""
    start, end, out_path = args
    chunk = _RECORDS[start:end]
    ds = datasets.Dataset.from_list(chunk, features=_FEATURES)
    embedded = embed_table_storage(ds.data.table)
    # Small row groups + page index keep each row group under HF dataset
    # viewer's 300 MB scan limit. With ~300 KB/row (embedded audio), 200
    # rows/group ≈ 60 MB.
    pq.write_table(embedded, out_path, row_group_size=200,
                   write_page_index=True)
    return end - start


def write_parquet_shards(
    records: List[Dict],
    features: datasets.Features,
    output_dir: Path,
    split_name: str,
    num_workers: int = 64,
    rows_per_shard: int = 2000,
) -> int:
    """Write ``records`` as sharded parquet under ``output_dir/data/``.

    Args:
        records: list of row dicts matching ``features``.
        features: target ``datasets.Features`` schema (must use ``Audio()``
            for any audio column so ``embed_table_storage`` inlines bytes).
        output_dir: dataset root; shards go under ``<output_dir>/data/``.
        split_name: e.g. ``"train"``, ``"train.clean.100"``.
        num_workers: parallel processes. Default 64; pinned by NFS bandwidth.
        rows_per_shard: target rows per shard. Smaller = more shards + more
            parallelism; larger = fewer files. 2000 is a reasonable midpoint
            for audio datasets where embedded shards are a few GB each.

    Returns:
        Total rows written.
    """
    global _RECORDS, _FEATURES

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    if n == 0:
        log.warning("[%s] no records; skipping", split_name)
        return 0

    # Stash on the module so forked workers see it without pickling.
    _RECORDS = records
    _FEATURES = features

    n_shards = max(1, (n + rows_per_shard - 1) // rows_per_shard)
    tasks = [
        (
            i * rows_per_shard,
            min((i + 1) * rows_per_shard, n),
            str(data_dir / f"{split_name}-{i:05d}-of-{n_shards:05d}.parquet"),
        )
        for i in range(n_shards)
    ]

    log.info(
        "[%s] writing %d rows in %d shards via %d workers -> %s",
        split_name, n, n_shards, num_workers, data_dir,
    )
    workers = min(num_workers, n_shards)
    with Pool(processes=workers, initializer=_init_worker) as pool:
        total = 0
        for written in pool.imap_unordered(_write_shard, tasks):
            total += written
    log.info("[%s] wrote %d rows", split_name, total)

    # Release the global so subsequent splits don't keep the previous list
    # alive across forks.
    _RECORDS = []
    _FEATURES = None  # type: ignore[assignment]
    return total
